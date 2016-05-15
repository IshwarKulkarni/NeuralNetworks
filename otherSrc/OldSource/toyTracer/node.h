#ifndef __NODE_INCLUDED__
#define __NODE_INCLUDED__

#include "toytracer.h"
#include "util.h"
#include "params.h" 
#include "constants.h"
#include <math.h>
#include <vector>
#include <algorithm>

#define MAX(a, b) ((a) < (b) ? (b) : (a))
#define _max(a, b, c) MAX( MAX(a,b), c)

using namespace std;

const double maxCost = 40;
const Interval zeroInterval = Interval(0);
//used for sorting
bool byX(Object* a, Object* b){
	return GetBox(*a).X.max < GetBox(*b).X.max;
}

bool byY(Object* a, Object* b){
	return GetBox(*a).Y.max < GetBox(*b).Y.max;			
}

bool byZ(Object* a, Object* b){
	return GetBox(*a).Z.max < GetBox(*b).Z.max;
}

struct Node{

	Node(){
			cost = 0.0; 
			leaf = true;
			aabb = AABB(); //initially everything is zero volume BB
			one = NULL;
			two = NULL;
			middle = NULL;
	}
	int insert(Object*);
	void printTree();	//only for debugging
	int retrunPrimitiveCount(); //only for debugging
	int returnNodeCount(); //only for debugging
	bool Intersect(const Ray, HitInfo&);
	void splitLeaf();

	bool leaf;
	double cost;	
	AABB aabb;
	vector<Object*> primitives;	// the midliers: objects that are temporarily held before node splits
	Node *one, *two, *middle;	//parent is unused right now
	int gen;	// generation
	
};

int Node::insert(Object* o){

	if(leaf){
		cost += o->Cost();	// the cost of the node is incremented
		primitives.push_back(o);
		aabb.Expand(GetBox(*o));

		if(cost>=maxCost)	// split criteria
			splitLeaf();
		return 0;
	}
	
	AABB box1;
	AABB box2;

	box1.Expand(one->aabb);
	box2.Expand(two->aabb);

	bool b1 = box1.Expand(GetBox(*o));	// !does o fully belong inside child one's AABB 
	bool b2 = box2.Expand(GetBox(*o));	// !does o fully belong inside child two's AABB 

	if(!(b1||b2) || (b1&&b2)){ // belongs to both or neither
		middle->insert(o);
		middle->aabb.Expand(GetBox(*o));
		return 3;
	}
	
	if(!b1) {	//fully belongs inside box one
		one->insert(o);
		one->aabb.Expand(GetBox(*o));
		return 1;
	}
	if(!b2){	//fully belongs inside box two
		two->insert(o);
		one->aabb.Expand(GetBox(*o));
		return 2;
	}
	//this case should not happen
	cout<<"should never print this\n";
	return -1;
}

void Node::printTree(){

	if(leaf){
		cout<<gen<<" : "<<primitives.size() <<endl;
		for(unsigned i = 0;i< primitives.size();i++)
			cout<< primitives[i]->MyName() << " @ " << (long) primitives[i];
		return;
	}
	cout<<"\n"<<gen<<" o: ";
	one->printTree();
	cout<<"\n"<<gen<<" m: ";
	middle->printTree();
	cout<<"\n"<<gen<<" t: ";
	two->printTree();
	cout<<"\nend of gen" <<gen <<endl;
	
}

int Node::retrunPrimitiveCount(){

	if( leaf)
		return primitives.size();
	
	return one->retrunPrimitiveCount() + middle->retrunPrimitiveCount() + two->retrunPrimitiveCount() ;

}

int Node::returnNodeCount(){

	if( leaf)
		return 1;
	
	return 3 + one->returnNodeCount()+ middle->returnNodeCount() +two->returnNodeCount();

}

bool Node::Intersect(const Ray r, HitInfo& hit){

	unsigned int pop= primitives.size();
	bool hitSomething = false;
	if(leaf){	// if leaf check the free primitivs an return
		for(unsigned i=0;i<pop;i++)
			if(primitives[i]->Hit(r,hit))
				hitSomething = true;
		return hitSomething;
	}


	if(middle->aabb.Hit(r,hit.distance))	// next check those that ended up in neither child nodes
		if( middle->Intersect(r,hit) )
			hitSomething = true;
		
	if(one->aabb.Hit(r,hit.distance))	// check the node one
		if(one->Intersect(r,hit))
			hitSomething = true;


	if(two->aabb.Hit(r,hit.distance) ) // finally check the node two
		if(two->Intersect(r,hit))
			hitSomething = true;

	return hitSomething;
}

void Node::splitLeaf(){

	leaf  = false;

	unsigned pop = primitives.size();	// population
	one = new Node();
	middle = new Node();
	two = new Node();
	
	one->gen = gen+1;			// set generations
	middle->gen = gen+1;
	two->gen = gen+1;
	
	// get maximum extent along each direction
	double max = _max(aabb.X.max - aabb.X.max , aabb.Y.max - aabb.Y.min, aabb.Z.max - aabb.Z.min);

	// sort by the axis along which the AABB is elongated the maximum
	if( max == (aabb.X.max - aabb.X.max )) 
		sort(primitives.begin(), primitives.end(),byX);
	else if( max == (aabb.Y.max - aabb.Y.max ))
		sort(primitives.begin(), primitives.end(),byY);
	else
		sort(primitives.begin(), primitives.end(),byZ);

	double tempCost = 0.0;
	unsigned i=0;
	for(  ;i<pop;i++){	// put roughly maxCost/2 weight of objects in node one.
		one->cost += primitives[i]->Cost();
		one->primitives.push_back(primitives[i]);
		one->aabb.Expand(GetBox(*primitives[i]));
		if(one->cost >= maxCost/2)
			break;
	}

	AABB box = one->aabb;
	for(unsigned j=i+1;j<pop;j++){ // handle the rest
		box = one->aabb;
		if( box.Expand(GetBox(*primitives[j]))){//this belongs to box one too, so goes to middle
			middle->cost += primitives[j]->Cost();
			middle->primitives.push_back(primitives[j]);	
			middle->aabb.Expand(GetBox(*primitives[j]));
		}
		else{	// does nto belong to one.
			two->cost += primitives[j]->Cost();
			two->primitives.push_back(primitives[j]);	
			two->aabb.Expand(GetBox(*primitives[j]));
		}
	}

	cost = 0.0;
	primitives.clear(); //free up memory
}
#endif