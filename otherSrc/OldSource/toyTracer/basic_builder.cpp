/***************************************************************************
* basic_builder.cpp                                                        *
*                                                                          *
* This file defines a basic "build" function, which reads a simple text    *
* description of a scene and the camera.                                   *
*                                                                          *
* History:                                                                 *
*   04/15/2010  Added containers & multi-line args with continuation symb. *
*   04/14/2010  Initialize plugins before reading the sdf file.            *
*   04/23/2006  The reader is now a "Builder" plugin.                      *
*   09/29/2005  Updated for 2005 class.                                    *
*   10/23/2004  Changed handling of default colors.                        *
*   10/16/2004  Minor updates (e.g. added more error messages).            *
*   10/06/2004  Added 3x4 matrix reader & support for aggregate objects.   *
*   04/04/2003  Initial coding.                                            *
*                                                                          *
***************************************************************************/
#include <fstream>
#include <string>
#include <list>
#include <vector>
#include "toytracer.h"
#include "util.h"
#include "params.h"
#include "constants.h"
#include "utils/exceptions.hxx"

using std::cout;
using std::cerr;
using std::endl;
using std::string;

namespace __basic_builder__ {

	struct stack_frame {
		bool      is_agg;
		ObjectSet *set;
		Shader    *shader;
		Envmap    *envmap;
		Material  *material;
	};

	struct basic_builder : Builder {
		basic_builder();
		virtual ~basic_builder() {}
		virtual bool BuildScene(string command, Camera &, Scene &);
		virtual Plugin *ReadString(const string &params);
		virtual string MyName() const { return "basic_builder"; }
		virtual bool Default() const { return true; }
	private: // Object-specific data and methods...
		bool Error(string);
		bool GetLine();
		int line_num;
		std::ifstream fileIn;
		string textline;
		std::list< stack_frame > stack;  // Maintain a stack of open structures.
		Material m_nil;
		Material m_def;

		bool Global() const { return stack.empty() || stack.front().set == NULL; }
		ObjectSet *CurrentSet() { return stack.empty() ? NULL : stack.front().set; }
		bool isAgg() { return stack.front().is_agg; }
		void SetAttribs(Object *);
		void Push(ObjectSet *, bool);
		void Pop();
		bool GetVertices();

		void CheckSceneIntegrity(Scene &scene)
		{
			THROW_IF(scene.object == NULL, FileIOException, "File does not describe an aggregate/a container")
		}
	};

	REGISTER_PLUGIN(basic_builder);


	basic_builder::basic_builder()
	{
		stack.clear();
		line_num = 0;
		const Color c_nil(-1, -1, -1);

		m_def.diffuse = Color::White;
		m_def.emission = Color::Black;
		m_def.specular = Color::White;
		m_def.ambient = Color::Black;
		m_def.reflectivity = Color::Black;
		m_def.translucency = Color::Black;
		m_def.ref_index = 0.0;
		m_def.Phong_exp = 0.0;

		m_nil.diffuse = c_nil;
		m_nil.emission = c_nil;
		m_nil.specular = c_nil;
		m_nil.ambient = c_nil;
		m_nil.reflectivity = c_nil;
		m_nil.translucency = c_nil;
		m_nil.ref_index = -1;
		m_nil.Phong_exp = -1;
	}

	void basic_builder::SetAttribs(Object *obj)
	{
		obj->shader = stack.front().shader;
		obj->envmap = stack.front().envmap;
		obj->material = new Material(*stack.front().material);
	}

	void basic_builder::Push(ObjectSet *set, bool is_agg)
	{
		stack.push_front(stack.front());
		stack.front().set = set;
		stack.front().is_agg = is_agg;
	}

	void basic_builder::Pop()
	{
		if (!stack.empty()) stack.pop_front();
	}

	Plugin *basic_builder::ReadString(const string &params)
	{
		ParamReader get(params);
		if (get["builder"] && get[MyName()]) return new basic_builder();
		return NULL;
	}

	// Check for any of the several characters that can signal the end of a line.
	static inline bool EndLine(char c)
	{
		return c == '\n' || c == '\r' || c == '\f' || c == '\0';
	}

	// If the line has nothing but white space before the end of line or
	// a comment symbol, then it can be ignored.
	static bool Skip(char *line)
	{
		for (char c = *line; !EndLine(c) && c != '#'; c = *++line)
		{
			if (c != ' ' && c != '\t') return false;
		}
		return true;
	}

	// If the last non-white symbol on the line is a continuation marker
	// (which is a backslash), then remove the marker and report that this
	// is a continuation.
	static bool Continuation(string &line)
	{
		for (int k = line.size() - 1; k >= 0; k--)
		{
			const char c(line[k]);
			if (c == '\\')
			{
				line.erase(k, 1); // Remove the continuation symbol.
				return true;
			}
			if (c != ' ' && c != '\t') break;  // Not a continuation.
		}
		return false;
	}

	// Allow data to be split over multiple lines by using a "continuation" character.
	// Each time we read a line whose last non-white character is a continuation marker,
	// append the line to the current line, remove the marker, and read another line.  Thus,
	// lines with continuation markers get concatenated into one long line before being
	// passed to the line parsers.
	bool basic_builder::GetLine()
	{
		static char buff[512];
		textline.clear();
		while (fileIn.getline(buff, 512))
		{
			line_num++; // Maintain original line count for accurate error reporting.
			if (!Skip(buff))
			{
				textline.append(buff);
				if (!Continuation(textline)) break;
			}
		}
		return !textline.empty();
	}

	// See if the material pointer is pointing at a black that is equivalent
	// to the one needed.  If so, leave it alone.  If not, create a new material
	// structure and reset the pointer to point to it.  This is how we share
	// material blocks among many objects.
	static Material *Copy(Material* &mat, const Material &material)
	{
		if (mat == NULL || !(*mat == material)) mat = new Material(material);
		return mat;
	}

	// Print an error message and current line number to standard error.
	bool basic_builder::Error(string msg)
	{
		cerr
			<< "\nError: " << msg
			<< ".  Line "
			<< line_num << ": "
			<< textline
			<< endl;
		return false;
	}

	// Try to share material blocks as much as possible by keeping a list
	// of all material blocks allocated thus far.  When a new one is requested,
	// first check if it has been allocated before.  Return a pointer to the
	// old one if so, and allocate a new one if not.
	static Material *GetMaterial(const Material *m)
	{
		static std::vector<Material*> cache;
		for (unsigned i = cache.size(); i > 0; i--)
		{
			if (*m == *cache[i - 1]) return cache[i - 1];
		}
		Material *new_m = new Material(*m);
		cache.push_back(new_m);
		return new_m;
	}

	// Copy attributes from the current stack frame to the given object.
	void FillMissingAttributes(Object *obj, const stack_frame &frame)
	{
		if (obj->shader == NULL) obj->shader = frame.shader;
		if (obj->envmap == NULL) obj->envmap = frame.envmap;
		if (obj->material == NULL) obj->material = GetMaterial(frame.material);
	}

	// Copy the objets from one ObjectSet to another.
	static void CopyContents(ObjectSet *into, ObjectSet *from, const stack_frame &frame)
	{
		if (into == NULL || from == NULL) return;
		from->Begin();
		for (;;)
		{
			Object *obj = from->GetChild();
			if (obj == NULL) break;
			FillMissingAttributes(obj, frame);
			into->AddChild(obj);
		}
	}

	bool basic_builder::GetVertices()
	{
		int i = 0;
		Vec3 v;
		while (GetLine())
		{
			ParamReader get(textline);
			if (get["end"]) break;
			while (get[v]) get.DefineIndex(i++, v);
		}
		return true;
	}

	// This is a very minimal scene description reader.  It assumes that
	// each line contains a complete entity: an object definition, or
	// a camera parameter, or a material parameter, etc.  (Blank lines, and
	// lines that begin with "#" are also allowed.)  It creates object instances,
	// links them together, and fills in the fields of the scene and camera as it
	// reads successive lines of the file.  Lines may be padded with blanks or
	// tabs to increase readability, or continued onto the next line by ending
	// them with a backslash

	bool basic_builder::BuildScene(string file_name, Camera &camera, Scene &scene)
	{
		Rasterizer *rst = NULL;  // The rasterizer to use.

		// Initialize counters so that we only count the objects created for building the scene,
		// not those in the list of plugins.

		Primitive::num_primitives = 0;
		Aggregate::num_aggregates = 0;
		Container::num_containers = 0;

		// Set reasonable defaults for all plugins in the absense of any specific
		// instructions in the scene description.
		scene.init();

		// Attempt to open the input file.
		line_num = 0;
		file_name += ".sdf";
		fileIn.open(file_name.c_str());
		THROW_IF(fileIn.fail(), FileIOException, "\nError: Could not open file %s", file_name.c_str());

		// Set some defaults.

		Material *mtl = new Material;
		mtl->diffuse = Color::White;
		mtl->emission = Color::Black;
		mtl->specular = Color::White;
		mtl->ambient = Color::Black;
		mtl->reflectivity = Color::Black;
		mtl->translucency = Color::Blue;
		mtl->ref_index = 0.0;
		mtl->Phong_exp = 0.0;

		stack_frame frame;
		frame.shader = scene.shader;
		frame.envmap = scene.envmap;
		frame.material = mtl;
		frame.set = NULL;

		stack.clear();
		stack.push_front(frame);

		camera.x_res = default_image_width;
		camera.y_res = default_image_height;
		camera.x_win = Interval(-1.0, 1.0);
		camera.y_win = Interval(-1.0, 1.0);

		// Process lines until the end of file is reached.
		// Print a warning for all lines that are unrecognizable.

		while (GetLine())
		{
			// Ask each registered object if it recognizes the line.  If it does, it will
			// create a new instance of itself and return a pointer to it.

			Plugin *the_plugin = Instance_of_Plugin(textline);

			if (the_plugin != NULL)
				switch (the_plugin->PluginType())
			{
				case shader_plugin:
					stack.front().shader = (Shader*)the_plugin;
					continue;

				case envmap_plugin:
				{
					Envmap *temp = (Envmap*)the_plugin;
					// If an environment map is set before any object is created, use it as
					// the background.
					if (Global()) scene.envmap = temp;
					else stack.front().envmap = temp;
					continue;
				}

				case rasterizer_plugin:
					THROW_IF(rst != NULL,FileIOException ,"More than one rasterizer specified");
					rst = (Rasterizer *)the_plugin;
					scene.rasterize = rst;
					continue;

				case primitive_plugin:
				{
					Object *obj = (Object*)the_plugin;
					SetAttribs(obj);
					if (isEmitter(obj)) scene.lights.push_back(obj);
					if (CurrentSet() != NULL)
					{
						// If there is an open aggregate object, add this primitive to it.
						CurrentSet()->AddChild(obj);
					}
					else
					{
						// If this is the first object encountered, use it as
						// the top-level object of the scene.
						if (scene.object == NULL) scene.object = obj;
					}
					continue;
				}

				case aggregate_plugin:
				{
					Aggregate *agg = (Aggregate*)the_plugin;
					SetAttribs(agg);
					if (Global())
					{
						// If this is the first top-level aggregate object encountered, use it as
						// the top-level object of the scene.
						if (scene.object == NULL) scene.object = agg;
					}
					Push(agg, true); // Make this the current aggregate.
					continue;
				}

				case container_plugin:
				{
					Container *cnt = (Container*)the_plugin;
					if (cnt->Size() == 0)
					{
						// This container is being defined.
						Push(cnt, false);
					}
					else
					{
						// Copy the contents of the container into the current aggregate or container.
						CopyContents(CurrentSet(), cnt, stack.front());
					}
					continue;
				}

				case null_plugin:
					// Do nothing.  These are used solely for their side effects.
					continue;
			}

			// Now look for all the other stuff...  materials, camera, lights, etc.

			ParamReader get(textline);

			Material *mat = stack.front().material;

			if (get["vertices"] && GetVertices()) continue;
			if (get["eye"] && get[camera.eye]) continue;
			if (get["lookat"] && get[camera.lookat]) continue;
			if (get["up"] && get[camera.up]) continue;
			if (get["vpdist"] && get[camera.vpdist]) continue;
			if (get["x_res"] && get[camera.x_res]) continue;
			if (get["y_res"] && get[camera.y_res]) continue;
			if (get["x_win"] && get[camera.x_win]) continue;
			if (get["y_win"] && get[camera.y_win]) continue;
			if (get["ambient"] && get[mat->ambient]) continue;
			if (get["diffuse"] && get[mat->diffuse]) continue;
			if (get["specular"] && get[mat->specular]) continue;
			if (get["emission"] && get[mat->emission]) continue;
			if (get["reflectivity"] && get[mat->reflectivity]) continue;
			if (get["translucency"] && get[mat->translucency]) continue;
			if (get["Phong_exp"] && get[mat->Phong_exp]) continue;
			if (get["ref_index"] && get[mat->ref_index]) continue;

			// Allow view plane normal to be specified instead of the "lookat" point.
			if (get["vpnormal"])
			{
				Vec3 normal;
				if (get[normal]) camera.lookat = camera.eye + normal;
				continue;
			}

			// Look for an end statement that closes the current aggregate.  This allows us to nest aggregates.

			if (get["end"])
			{
				if (Global()) return Error("end statement outside an aggregate or container");

				ObjectSet *ended_set = CurrentSet();
				bool ended_isagg = isAgg();

				Pop(); // Go back to previous object set.

				ObjectSet *outer_set = CurrentSet();
				bool outer_isagg = isAgg();

				// Handle the object set that just ended differently, depending on whether
				// it was an Aggregate object or a Container.

				if (ended_isagg)
				{
					Aggregate *ended_agg = (Aggregate*)ended_set;
					ended_agg->Close();
					if (outer_set != NULL && outer_isagg)
					{
						Aggregate *outer_agg = (Aggregate*)outer_set;
						outer_agg->AddChild(ended_agg);
					}
					if (outer_set != NULL && !outer_isagg)
					{
						Container *outer_cnt = (Container*)outer_set;
						outer_cnt->AddChild(ended_agg);
					}
				}
				else // The object set that just ended is a container.
				{
					Container *ended_cnt = (Container*)ended_set;
					ended_cnt->Close();
					if (outer_set != NULL && outer_isagg)
					{
						Aggregate *outer_agg = (Aggregate*)outer_set;
						CopyContents(outer_agg, ended_cnt, stack.front());
					}
					if (outer_set != NULL && !outer_isagg)
					{
						Container *outer_cnt = (Container*)outer_set;
						CopyContents(outer_cnt, ended_cnt, stack.front());
					}
				}

				continue;
			}

			// If no object is defined at this point, it's an error.

			THROW_IF(Global(), FileIOException, "While reading scene file; No object defined");

			// If nothing matched, it's an error.  Print an error message and continue processing.

			Error("Unrecognized command");
		}

		THROW_IF(CurrentSet() != NULL, FileIOException, "Set object is missing an 'end' statement");
		CheckSceneIntegrity(scene);
		fileIn.close();
		return true;
	}

} // namespace __basic_builder__
