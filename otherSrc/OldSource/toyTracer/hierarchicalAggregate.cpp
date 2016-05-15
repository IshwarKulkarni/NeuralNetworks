#include "toytracer.h"
#include "util.h"
#include "params.h" 
#include "constants.h"
#include "node.h"

using namespace std;

namespace __hierarchical_aggregate__ {  

	struct hierarchicalAggregate:Aggregate{

		hierarchicalAggregate(){root = new Node();}
		~hierarchicalAggregate() { if( root != NULL ) delete root;root = NULL;}
		virtual bool Intersect( const Ray &ray, HitInfo & ) const;
		virtual bool Inside( const Vec3 & ) const;
		virtual Interval GetSlab( const Vec3 & ) const;
		virtual Plugin *ReadString( const std::string &params );
		virtual std::string MyName() const { return "hierarchicalAggregate"; }
		virtual void Close();
		//virtual void AddChild( Object * );
		AABB aabb;

		Node *root;		//a node of tree described in the other file
	};

	REGISTER_PLUGIN( hierarchicalAggregate );
	
	Plugin *hierarchicalAggregate::ReadString( const std::string &params ){
		ParamReader get( params );
		if( get["begin"] && get[MyName()] )
			return new hierarchicalAggregate();
		return NULL;
    }
		
	Interval hierarchicalAggregate::GetSlab(const Vec3& a )const {
		Interval I;
		for(unsigned i=0;i< children.size();i++){
			I <<children[i]->GetSlab(a);
		}
		return I;
	}

	bool hierarchicalAggregate::Inside(const Vec3& a )const {

		// pretty inefficient
		for(unsigned i=0;i<  children.size();i++)
			if( children[i]->Inside(a)) return true;
		return false;

	}

	bool hierarchicalAggregate::Intersect( const Ray &ray, HitInfo &hit)const {
		// call intersect of root, everything else is handled from there.
		return root->Intersect(ray,hit);
	}

	void hierarchicalAggregate::Close(){
		unsigned pop = children.size();// population
		
		root->gen = 0;	
		std::cout<<"To insert : " <<pop<<std::endl;

		cout<< " \nInserting..";
		for(unsigned i=0;i< pop;i++){
			root->insert(children[i]); 
		}	
		cout<<"Done.. Inserted average "<< (float)root->retrunPrimitiveCount()/(float)root->returnNodeCount()<<endl;
		//root->printTree();
	}

}