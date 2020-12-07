// Liam Bessell, 11/20/20, CSCE 489 Computer Animation, Dr. Sueda

#include "Octree.h"
#include "Particle.h"
#include <limits>
#include <iostream>

OctreeNode::OctreeNode()
{
    children = std::vector<OctreeNode*>();
    particle = NULL;
    centerOfMass << 0, 0, 0;
    mass = 0;
    position << 0, 0, 0;
    length = 0;
}

OctreeNode::OctreeNode(Eigen::Vector3d position_, double length_)
{
    children = std::vector<OctreeNode*>();
    particle = NULL;
    centerOfMass << 0, 0, 0;
    mass = 0;
    position = position_;
    length = length_;
}

void OctreeNode::Draw(bool altColor)
{
    // Draw the bounding box for the node
    if (altColor)
    {
        glColor3f(1, 0, 0);
    }
    else
    {
        glColor3f(0, 1, 0);
    }

    glLineWidth(1);
    glBegin(GL_LINES);
    
    glVertex3f(position(0), position(1), position(2));
    glVertex3f(position(0)+length, position(1), position(2));
    glVertex3f(position(0), position(1), position(2));
    glVertex3f(position(0), position(1)+length, position(2));
    glVertex3f(position(0), position(1), position(2));
    glVertex3f(position(0), position(1), position(2)+length);

    glVertex3f(position(0)+length, position(1)+length, position(2)+length);
    glVertex3f(position(0), position(1)+length, position(2)+length);
    glVertex3f(position(0)+length, position(1)+length, position(2)+length);
    glVertex3f(position(0)+length, position(1), position(2)+length);
    glVertex3f(position(0)+length, position(1)+length, position(2)+length);
    glVertex3f(position(0)+length, position(1)+length, position(2));

    glVertex3f(position(0), position(1), position(2)+length);
    glVertex3f(position(0)+length, position(1), position(2)+length);
    glVertex3f(position(0), position(1), position(2)+length);
    glVertex3f(position(0), position(1)+length, position(2)+length);

    glVertex3f(position(0), position(1)+length, position(2));
    glVertex3f(position(0), position(1)+length, position(2)+length);
    glVertex3f(position(0), position(1)+length, position(2));
    glVertex3f(position(0)+length, position(1)+length, position(2));

    glVertex3f(position(0)+length, position(1), position(2));
    glVertex3f(position(0)+length, position(1)+length, position(2));
    glVertex3f(position(0)+length, position(1), position(2));
    glVertex3f(position(0)+length, position(1), position(2)+length);

    glEnd();
}

Octree::Octree()
{
    root = NULL;
}

Octree::Octree(std::vector<std::shared_ptr<Particle>>& particles)
{
    // Bounding cube dimensions, O(n)
    double maxDimension = std::numeric_limits<double>::min();
    double minDimension = std::numeric_limits<double>::max();
    for (auto p : particles)
    {
        double x = p->getPosition()(0);
        if (x > maxDimension)
        {
            maxDimension = x + 1e-1;
        }
        if (x < minDimension)
        {
            minDimension = x - 1e-1;
        }

        double y = p->getPosition()(1);
        if (y > maxDimension)
        {
            maxDimension = y + 1e-1;
        }
        if (y < minDimension)
        {
            minDimension = y - 1e-1;
        }

        double z = p->getPosition()(2);
        if (z > maxDimension)
        {
            maxDimension = z + 1e-1;
        }
        if (z < minDimension)
        {
            minDimension = z - 1e-1;
        }
    }

    // Construct the tree
    double length = maxDimension - minDimension;
    Eigen::Vector3d position;
    position << minDimension, minDimension, minDimension;
    root = new OctreeNode(position, length);
    for (auto p : particles)
    {
        Insert(p, root);
    }

    nodesToDraw = std::vector<OctreeNode*>();
}

Octree::~Octree()
{
    // Traverse the tree and delete each node
    int nonEmptyLeafCount = 0;
    PostOrderDestruct(root, &nonEmptyLeafCount);
    //std::cout << "Non-empty leaf count: " << nonEmptyLeafCount << std::endl;
}

void Octree::PostOrderDestruct(OctreeNode* node, int* nonEmptyLeafCount)
{
    if (node->IsLeaf() && !node->IsEmpty())
    {
        (*nonEmptyLeafCount)++;
    }
    else
    {
        for (int i = 0; i < node->children.size(); i++)
        {
            PostOrderDestruct(node->children[i], nonEmptyLeafCount);
        }
    }

    delete node;
}

void Octree::ComputeAllCentersOfMass()
{
    double mass = 0;
    Eigen::Vector3d centerOfMass;
    centerOfMass << 0, 0, 0;
    ComputeCenterOfMass(root, &mass, &centerOfMass);
}

void Octree::ComputeAllForces(std::vector<std::shared_ptr<Particle>>& particles, double h, Eigen::MatrixXd* forceMat, const double G, const double e2, const double theta, std::shared_ptr<Particle> selectedParticle)
{
    for (int i = 0; i < particles.size(); i++)
    {
        Eigen::Vector3d force;
        force << 0, 0, 0;
        if (particles[i] == selectedParticle)
        {
            ComputeForceOnParticle(particles[i], root, &force, G, e2, theta, true);
        }
        else
        {
            ComputeForceOnParticle(particles[i], root, &force, G, e2, theta, false);
        }
        forceMat->block<3, 1>(0, i) = force;
    }
}

void Octree::ComputeCenterOfMass(OctreeNode* node, double* mass_, Eigen::Vector3d* centerOfMass_)
{
    double totMass = 0;
    Eigen::Vector3d totCenterOfMass;
    totCenterOfMass << 0, 0, 0;

    // Post-order traversal
    if (node->IsLeaf() && !node->IsEmpty())
    {
        totMass = node->particle->getMass();
        totCenterOfMass = node->particle->getPosition();
    }
    else if (!node->IsLeaf())
    {
        for (int i = 0; i < node->children.size(); i++)
        {
            double mass = 0;
            Eigen::Vector3d centerOfMass;
            centerOfMass << 0, 0, 0;

            ComputeCenterOfMass(node->children[i], &mass, &centerOfMass);

            totMass += mass;
            totCenterOfMass += mass*centerOfMass;
        }

        if (totMass > 0)
        {
            totCenterOfMass /= totMass;
        }
    }

    node->mass = totMass;
    node->centerOfMass = totCenterOfMass;
    *mass_ = node->mass;
    *centerOfMass_ = node->centerOfMass;
}

void Octree::ComputeForceOnParticle(std::shared_ptr<Particle> particle, OctreeNode* node, Eigen::Vector3d* force_, const double G, const double e2, const double theta, bool drawNodes)
{
    if (node->IsLeaf() && !node->IsEmpty())
    {
        if (drawNodes)
        {
            nodesToDraw.push_back(node);
        }
        
        if (node->particle == particle)
        {
            return;
        }

        Eigen::Vector3d r = node->centerOfMass - particle->getPosition();
        double rNorm = r.norm();
        *force_ = ((G * particle->getMass() * node->mass) / std::pow(rNorm * rNorm + e2, 1.5)) * r;
    }
    else if (!node->IsLeaf())
    {
        Eigen::Vector3d r = node->centerOfMass - particle->getPosition();
        double D = node->length;
        double rNorm = r.norm();
        if (D/rNorm < theta)
        {
            if (drawNodes)
            {
                nodesToDraw.push_back(node);
            }

            *force_ = ((G * particle->getMass() * node->mass) / std::pow(rNorm * rNorm + e2, 1.5)) * r;
        }
        else
        {
            for (int i = 0; i < node->children.size(); i++)
            {
                Eigen::Vector3d force;
                force << 0, 0, 0;
                ComputeForceOnParticle(particle, node->children[i], &force, G, e2, theta, drawNodes);
                *force_ += force;
            }
        }
    }
}

void Octree::Draw(std::shared_ptr<Particle> selectedParticle)
{
    for (int i = 0; i < nodesToDraw.size(); i++)
    {
        if (nodesToDraw[i]->particle == selectedParticle)
        {
            nodesToDraw[i]->Draw(true);
        }
        else
        {
            nodesToDraw[i]->Draw(false);
        }
    }
}

void Octree::Draw(OctreeNode* node, const bool drawEmptyLeaves)
{
    if (node->IsLeaf())
    {
        if (!node->IsEmpty())
        {
            node->Draw(false);
        }
        else if (drawEmptyLeaves)
        {
            node->Draw(false);
        }
    }
    else
    {
        for (int i = 0; i < node->children.size(); i++)
        {
            Draw(node->children[i], drawEmptyLeaves);
        }
    }
}

void Octree::Insert(std::shared_ptr<Particle> particle, OctreeNode* node)
{
    if (!node->IsLeaf())
    {
        // Find child that the particle lies in
        for (int i = 0; i < 8; i++)
        {
            if (IsPointInsideBoundingBox(particle->getPosition(), node->children[i]->position, node->children[i]->length))
            {
                Insert(particle, node->children[i]);
                break;
            }
        }
    }
    else if (!node->IsEmpty())
    {
        // Subdivide the node into octants
        double childLength = node->length / 2.0;
        Eigen::Vector3d childPosition;
        
        for (int i = 0; i < 2; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                for (int k = 0; k < 2; k++)
                {
                    childPosition << node->position(0) + i*childLength,
                                     node->position(1) + j*childLength,
                                     node->position(2) + k*childLength;
                    node->children.push_back(new OctreeNode(childPosition, childLength));
                }
            }
        }

        // Find child that the original particle lies in
        for (int i = 0; i < 8; i++)
        {
            if (IsPointInsideBoundingBox(node->particle->getPosition(), node->children[i]->position, node->children[i]->length))
            {
                Insert(node->particle, node->children[i]);
                break;
            }
        }

        // Find child that the new particle lies in
        for (int i = 0; i < 8; i++)
        {
            if (IsPointInsideBoundingBox(particle->getPosition(), node->children[i]->position, node->children[i]->length))
            {
                Insert(particle, node->children[i]);
                break;
            }
        }

        node->particle = NULL;
    }
    else
    {
        node->particle = particle;
    }
}

bool IsPointInsideBoundingBox(Eigen::Vector3d point, Eigen::Vector3d boxPosition, double boxLength)
{
    if (point(0) >= boxPosition(0) && point(0) < boxPosition(0) + boxLength
     && point(1) >= boxPosition(1) && point(1) < boxPosition(1) + boxLength
     && point(2) >= boxPosition(2) && point(2) < boxPosition(2) + boxLength)
    {
        return true;
    }

    return false;
}