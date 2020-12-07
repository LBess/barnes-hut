// Liam Bessell, 11/20/20, CSCE 489 Computer Animation, Dr. Sueda

#pragma once
#ifndef _OCTREE_H_
#define _OCTREE_H_


#define EIGEN_DONT_ALIGN_STATICALLY
#include <Eigen/Dense>

#include <vector>
#include <memory>

class Particle;

class OctreeNode
{
public:
    OctreeNode();
    OctreeNode(Eigen::Vector3d position_, double length_);
    bool IsLeaf() { return children.size() == 0 ? true : false; }
    bool IsEmpty() { return particle == NULL ? true : false; }
    void Draw(bool altColor);

    std::vector<OctreeNode*> children;
    std::shared_ptr<Particle> particle;
    Eigen::Vector3d centerOfMass;
    double mass;
    // Bounding box position. Minimum corner (since all boxes are orthogonal)
    Eigen::Vector3d position;
    // Bounding box dimensions
    double length;
};

class Octree
{
public:
    Octree();
    Octree(std::vector<std::shared_ptr<Particle>>& particles);
    ~Octree();
    void PostOrderDestruct(OctreeNode* node, int* nonEmptyLeafCount);
    void ComputeAllCentersOfMass();
    void ComputeAllForces(std::vector<std::shared_ptr<Particle>>& particles, double h, Eigen::MatrixXd* forceMat, const double G, const double e2, const double theta, std::shared_ptr<Particle> selectedParticle = NULL);
    void Draw(std::shared_ptr<Particle> selectedParticle);
    void Draw(OctreeNode* node, const bool drawEmptyLeaves);
    OctreeNode* GetRoot() { return root; }

private:
    void Insert(std::shared_ptr<Particle> particle, OctreeNode* node);
    void ComputeCenterOfMass(OctreeNode* node, double* mass_, Eigen::Vector3d* centerOfMass_);
    void ComputeForceOnParticle(std::shared_ptr<Particle> particle, OctreeNode* node, Eigen::Vector3d* force_, const double G, const double e2, const double theta, bool drawNodes);

    OctreeNode* root;
    std::vector<OctreeNode*> nodesToDraw;
};

bool IsPointInsideBoundingBox(Eigen::Vector3d point, Eigen::Vector3d boxPosition, double boxLength);

#endif
