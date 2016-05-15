/***************************************************************************
* abvh.cpp   (aggregate object plugin)                                     *
*                                                                          *
* The abvh (Automatic Bounding Volume Hierarchy) creates a bounding        *
* volume hierarchy using branch-and-bound to find an optimal hierarchy     *
* with respect to a probabilistic objective function.  The resulting       *
* hierarchy depends somewhat on the order in which the objects are added,  *
* although with respect to a given ordering, the hierarchy will be         *
* essentially optimal according to the objective function.                 *
*                                                                          *
* History:                                                                 *
*   04/25/2010  Added NextObject method to node_iterator.                  *
*   04/16/2010  Added randomized insertion order.                          *
*   10/09/2005  Ported from a previous ray tracer.                         *
*                                                                          *
***************************************************************************/
#include <string>
#include "toytracer.h"
#include "util.h"
#include "params.h"
#include "constants.h"

using std::string;

// No code outside of this module will refer to any of the types defined here,
// as all access will be through virtual methods.  Thus, we can enclose the
// entire module in a unique namespace simply to ensure that there will be no
// name collisions with system header files.

namespace __abvh_aggregate__ {

	// The node struct forms all of the nodes in the bounding volume hierarchy,
	// which can be a tree with arbitrary branching (i.e. internal nodes can
	// have any number of children).
	struct node {
		inline  node();
		inline ~node();
		inline  double Cost() const { return ec + sa * aic; }
		inline  bool   Leaf() const { return child == NULL; }
		double  AdjustedInternalCost() const;
		void    AddChild(node *);
		AABB    bbox;    // The bounding box containing all the children.
		double  ec;      // External cost of this object.
		double  sa;      // Surface area -- can be zero for a primitive.
		double  aic;     // Adjusted internal cost.
		double  sec;     // Sum of external costs of children.
		double  saic;    // Sum of adjusted internal costs of children.
		node   *sibling; // The next child of the node that contains this one. 
		node   *parent;  // The node one level up, or NULL if this is the root.
		node   *child;   // An object or volume nested inside this one.
		Object *object;  // The actual object bounded.
	};

	// Initialize all the fields of the new abvh object.
	inline node::node()
	{
		ec = 0.0;
		sa = 0.0;
		aic = 0.0;
		sec = 0.0;
		saic = 0.0;
		sibling = NULL;
		parent = NULL;
		child = NULL;
		object = NULL;
		bbox = AABB::Null();
	}

	// Recursively delete every object and node in the tree from the given
	// node down.
	inline node::~node()
	{
		if (object != NULL) delete object;
		if (child != NULL) delete child;
		if (sibling != NULL) delete sibling;
		object = NULL;
		child = NULL;
		sibling = NULL;
	}

	// The node_iterator struct is the mechanism for walking through a bounding
	// volume hierarchy, either accessing all the volumes (in-order traversal),
	// or skipping sub-trees.
	struct node_iterator {
		inline node_iterator(const node *r = NULL) { init(r); }
		inline ~node_iterator() {}
		inline void init(const node *r) { curr = r; }
		inline bool IsLeaf() const { return curr->child == NULL; }
		inline const node *SkipChildren(); // Next volume, ignoring children.
		inline const node *Next(); // Down if possible, then to sibling, or up.
		inline const node *Curr() const { return curr; }
		inline Object *NextObject();
	private: const node *curr;
	};

	inline const node *node_iterator::SkipChildren()
	{
		// Advance the "current" pointer (curr) to the next unvisited node
		// at the same level or higher.  Since the parent has already been
		// visited, when we move "up" we must also move to the next "sibling".
		while (curr != NULL)
		{
			if (curr->sibling != NULL)
			{
				curr = curr->sibling;
				break;
			}
			curr = curr->parent;
		}
		return curr;
	}

	inline const node *node_iterator::Next()
	{
		// Find the next child in the hierarchy by first going "down" to
		// a child node, if possible.  When there is no child node, try
		// moving to a sibling.  If that fails, move up to the parent first
		// and then to a sibling (i.e. try to find an uncle).
		if (curr->child == NULL) SkipChildren();
		else curr = curr->child;
		return curr;
	}

	inline Object *node_iterator::NextObject()
	{
		// Step through the nodes of the hierarchy until we find a leaf.
		// Return its object pointer.
		Next();
		while (curr != NULL && !curr->Leaf()) Next();
		return curr == NULL ? NULL : curr->object;
	}

	enum permutation_type {
		no_perm,             // Insert elements in original order.
		random_perm,         // Use "rand" to randomize insertion order.
		heuristic_perm       // Use a simple heuristic to shuffle order.
	};

	struct abvh : Aggregate {
		abvh(permutation_type = heuristic_perm, unsigned seed = 0);
		virtual ~abvh();
		virtual bool Intersect(const Ray &ray, HitInfo &) const;
		virtual bool Inside(const Vec3 &) const;
		virtual Interval GetSlab(const Vec3 &) const;
		virtual Plugin *ReadString(const string &params);
		virtual string MyName() const { return "abvh"; }
		virtual void Begin() const { tree_walker.init(root); }
		virtual Object *GetChild() const { return tree_walker.NextObject(); }
		virtual void Close();
		virtual double Cost() const;

	private: // Object-specific data and methods...
		static bool Branch_and_Bound(node*, node*, node*&, double&);
		void Insert(Object *, double relative_cost = 1.0);
		node *root;
		permutation_type ptype;
		unsigned seed;
		mutable node_iterator tree_walker;
	};

	REGISTER_PLUGIN(abvh);

	abvh::abvh(permutation_type ptype_, unsigned seed_)
	{
		root = NULL;
		ptype = ptype_;
		seed = seed_;
	}

	Plugin *abvh::ReadString(const string &params)
	{
		// The abvh aggregate accepts some optional parameters following
		// the keyword "begin" and the name "abvh".
		ParamReader get(params);
		if (get["begin"] && get[MyName()])
		{
			while (!get.isEmpty())
			{
				if (get["randomize"]) { ptype = random_perm;    continue; }
				if (get["shuffle"]) { ptype = heuristic_perm; continue; }
				if (get["in-order"]) { ptype = no_perm;        continue; }
				if (get["seed"]) { get[seed];              continue; }
				get.Warning("unprocessed parameters to " + MyName());
				break;
			}
			return new abvh(ptype, seed);
		}
		return NULL;
	}

	// When the object is closed, add each of the child objects to the hierarchy.
	// Waiting until we have all the objects allows us to radomize the order of
	// object insertion, which can greatly affect the quality of the resulting
	// bounding volume hierarchy.
	void abvh::Close()
	{
		int *perm(NULL);
		const unsigned n(Size());

		// Change the order of insertion by using either a "random" permutation
		// or a "deterministic" permutation if either was requested.
		if (ptype == random_perm)
		{
			perm = new int[n];
			random_permutation(n, perm, seed);
		}
		else if (ptype == heuristic_perm)
		{
			perm = new int[n];
			non_random_permutation(n, perm, seed);
		}

		for (unsigned i = 0; i < n; i++)
		{
			const int k(perm == NULL ? i : perm[i]);
			Object *obj(children[k]);
			// Take into account the estimated cost of intersecting a ray with
			// this object as we insert it into the existing bvh.
			Insert(obj, obj->Cost());
		}

		if (root != NULL)
		{
			// See how well the optimization did...
			// std::cout << "\nEstimated cost of tree = " << root->Cost() << std::endl;
		}

		// Dispose of the permutation and the original list of children.
		if (perm != NULL) delete[] perm;
		children.clear();
	}

	// Recursively delete the entire tree by deleting the root.
	abvh::~abvh()
	{
		if (root != NULL) delete root;
		root = NULL;
	}

	bool abvh::Intersect(const Ray &ray, HitInfo &hitinfo) const
	{
		// Walk the bounding volume hierarchy intersecting, descending down a
		// branch if and only if the ray intersects the volume.  When a leaf is
		// reached, intersect the ray with the object found there.
		bool found_a_hit(false);
		double closest_hit(Infinity);
		node_iterator iter(root);
		while (iter.Curr() != NULL)
		{
			if (iter.IsLeaf())
			{
				const Object *obj(iter.Curr()->object);
				// If we've reached a node in the tree that is a leaf, it's
				// a primitive object, so we must test it against the ray.
				if (obj != ray.ignore && obj->Hit(ray, hitinfo))
				{
					// We've hit an object, but continue looking unless this ray is for
					// testing visibility only, in which case we do not need the closest hit.
					if (ray.type == visibility_ray) return true;
					found_a_hit = true;
					closest_hit = hitinfo.distance;
				}
				iter.Next();
			}
			else if (iter.Curr()->bbox.Hit(ray, closest_hit))
			{
				// If the node is not a leaf, it's a bounding box, so we test
				// the ray against it.  If it's hit, we descend into the
				// children of the node.
				iter.Next();
			}
			else
			{
				// If the bounding box is not hit, we ignore the children
				// (since the ray cannot possibly hit them) and continue
				// processing nodes at the current or higher level.
				iter.SkipChildren();
			}
		}
		return found_a_hit;
	}

	// Test to see if the point P is "inside" the object.  For an aggregate object
	// this is done by asking each of the child objects.
	bool abvh::Inside(const Vec3 &P) const
	{
		node_iterator iter(root);
		while (iter.Curr() != NULL)
		{
			if (iter.IsLeaf() && iter.Curr()->object->Inside(P)) return true;
			iter.Next();
		}
		return false;
	}

	// Return an interval that bounds the object in the given direction.  For an
	// aggregate this is done by consulting all the child objects and combining
	// the intervals they provide.
	Interval abvh::GetSlab(const Vec3 &v) const
	{
		Interval I;
		node_iterator iter(root);
		while (iter.Curr() != NULL)
		{
			// For each leaf node encountered, expand the interval to
			// include the interval of the child object.
			if (iter.IsLeaf()) I << iter.Curr()->object->GetSlab(v);
			iter.Next();
		}
		return I;
	}

	// This function adds an object to an existing bounding volume hierarchy.
	// It first constructs a bounding box for the new object, then finds the
	// optimal place to insert it into the hierachy using branch-and-bound, and
	// finally inserts it at the determined location, updating all the costs
	// associated with the nodes in the hierarchy as a side-effect.
	void abvh::Insert(Object *obj, double relative_cost)
	{
		const AABB box(GetBox(*obj));
		const double surfarea(SurfaceArea(box));

		// If the bounding box has zero surface area, then this object is a point
		// or some other thing that cannot be ray traced.  So skip it in order to
		// keep the bounding box smaller.
		if (surfarea <= 0.0) return;

		double bound(0.0);
		node *n(new node);

		n->bbox = box;
		n->object = obj;
		n->sa = surfarea;
		n->ec = 1.0;
		n->sec = relative_cost;
		n->aic = relative_cost * n->sa;
		n->saic = 0.0; // There are no child volumes yet.

		// If this is the first node being added to the hierarchy,
		// it becomes the root.
		if (root == NULL) { root = n; return; }

		// Look for the optimal place to add the new node.
		// The branch-and-bound procedure will figure out where
		// to put it so as to cause the smallest increase in the
		// estimated cost of the tree.
		bound = Infinity;
		node *insert_here = NULL;
		Branch_and_Bound(root, n, insert_here, bound);
		// Now actually insert the node in the optimal place.
		insert_here->AddChild(n);
	}

	// Estimate the average cost of intersecting a ray with this object.
	double abvh::Cost() const
	{
		return (root == NULL) ? 1.0 : root->Cost();
	}

	// The "Adjusted Internal Cost" of a bounding volume is the product of
	// its surface area and its internal cost.  By definition, this is zero
	// for primitive objects.
	double node::AdjustedInternalCost() const
	{
		// Both "Internal Cost" and "Adjusted Internal Cost" of a
		// leaf are zero by definition.
		if (child == NULL) return 0.0;
		double sum_ec(0);
		double sum_aic(0);
		for (const node *c = child; c != NULL; c = c->sibling)
		{
			sum_ec += c->ec;
			sum_aic += c->AdjustedInternalCost();
		}
		return sa * sum_ec + sum_aic;
	}

	// This method determines the optimal location to insert a new node
	// into the hierarchy; that is, the place that will create the smallest
	// increase in the expected cost of intersecting a random ray with the
	// new hierarchy.  This is accomplished with branch-and-bound, which finds
	// the same solution as a brute-force search (i.e. attempting to insert
	// the new object at *each* possible location in the hierarchy), but
	// does so efficiently by pruning large portions of the tree.
	bool abvh::Branch_and_Bound(
		node  *the_root,     // Root of the tree to be modified.
		node  *the_node,     // The node to add to the hierarchy.
		node  *&best_parent, // The best node to add it to.
		double &bound)      // Smallest bound achieved thus far.

	{
		double a_delta;  // AIC increase due to Root bbox expansion. 
		double r_delta;  // AIC increase due to new child of Root.   
		double c_bound;  // bound - a_delta, or less.               

		// Compute the increase in the bounding box of the root due 
		// to adding the new object.  This is used in computing     
		// the new area cost no matter where it ends up.            
		AABB root_box(the_root->bbox);
		bool expanded(root_box.Expand(the_node->bbox));

		// Compute the common sub-expression of the new area cost  
		// and the increment over this that would occur if we     
		// made the new object a child of the_root.                    
		if (expanded)
		{
			const double new_area(SurfaceArea(root_box));
			a_delta = (new_area - the_root->sa) * the_root->sec;
			c_bound = bound - a_delta;
			if (c_bound <= 0.0) return false; // Early cutoff. 
			r_delta = new_area * the_node->ec + the_node->aic;
		}
		else
		{
			a_delta = 0.0;
			c_bound = bound;
			r_delta = the_root->sa * the_node->ec + the_node->aic;
		}

		// See if adding the new node directly to the root of this tree
		// achieves a better bound than has thus far been obtained (via
		// previous searches).  If so, update all the parameters to reflect
		// this better choice.
		bool improved(false);
		if (r_delta < c_bound)
		{
			bound = a_delta + r_delta;  // This is < *bound. 
			best_parent = the_root;
			c_bound = r_delta;
			improved = true;
		}

		// Compute the smallest increment in total area cost (over 
		// "common") which would result from adding the new node 
		// to one of the children of the_root.  If any of them obtains
		// a better bound than achieved previously, then the "best_parent"
		// and "bound" parameters will be updated as a side effect.
		for (node *c = the_root->child; c != NULL; c = c->sibling)
		{
			if (Branch_and_Bound(c, the_node, best_parent, c_bound))
			{
				bound = c_bound + a_delta;
				improved = true;
			}
		}
		return improved;
	}

	// This function inserts the given child node into the bounding volume
	// hierarchy and adjusts all the associated costs that are stored in
	// the tree.
	void node::AddChild(node *new_child)
	{
		AABB new_volume(new_child->bbox);

		if (child == NULL)
		{
			// The current node is a leaf, so we must convert it into
			// an internal node.  To do this, copy its current contents into
			// a new node, which will become a child of this one.
			child = new node;
			child->bbox = bbox;
			child->object = object;
			child->parent = this;

			// Fill in all the fields of the current node, which has just changed
			// into an internal node with a single child.
			ec = 1;
			sa = child->sa;
			sec = child->ec;  // There is only one child.
			saic = child->aic; // There is only one child.
			aic = sa * sec + saic;
		}

		// Splice the new child into the linked list of children.  The children are
		// linked via the "sibling" pointer, and all children point to the parent.
		new_child->sibling = child;
		new_child->parent = this;
		child = new_child;

		// Update the summed external cost and the summed AIC as a result
		// of adding the new child node.  These do not depend on surface
		// area, so we needn't consider any expansion of the bounding volume.
		sec += new_child->ec;
		saic += new_child->aic;

		// Now take bounding volume expansion into account due to the new child.
		bool expanded(false);
		if (bbox.Expand(new_volume))
		{
			expanded = true;
			sa = SurfaceArea(bbox);
		}

		// Compute new area cost & how much it increased due to new child.
		double old_aic(aic);
		aic = sa * sec + saic;
		double increment(aic - old_aic);

		// Propagate the information upward in two phases.  The first phase
		// deals with volumes that get expanded.  Once a volume is reached that does
		// not expand, there will be no more expansion all the way to the root.
		node *n(this);
		while (expanded)
		{
			expanded = false;
			n = n->parent;
			if (n != NULL && n->bbox.Expand(new_volume))
			{
				expanded = true;
				old_aic = n->aic;
				n->sa = SurfaceArea(n->bbox);
				n->saic += increment;
				n->aic = n->sa * n->sec + n->saic;
				increment = n->aic - old_aic;
			}
		}

		// From this point up to the root there will be no more expansion.
		// However, we must still propagate information upward.
		while (n != NULL)
		{
			n->saic += increment;
			n->aic += increment;
			n = n->parent;
		}
	}

} // namespace __abvh_aggregate__



