// Liam Bessell, 11/20/20, CSCE 489 Computer Animation, Dr. Sueda

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Camera.h"
#include "GLSL.h"
#include "Program.h"
#include "MatrixStack.h"
#include "Octree.h"
#include "Particle.h"
#include "Texture.h"
#include "Octree.h"
#include "MyTimer.h"

using namespace std;
using namespace Eigen;

bool keyToggles[256] = {false};

GLFWwindow *window;							// Main application window
string RESOURCE_DIR = "";					// Where the resources are loaded from

shared_ptr<Program> progSimple;
shared_ptr<Program> prog;
shared_ptr<Camera> camera;
vector< shared_ptr<Particle> > particles;
shared_ptr<Texture> texture;
double t, h, e2;

bool paused = false;
bool barnesHut = true;
bool drawOctree = false;
bool drawEmptyLeaves = false;
Eigen::MatrixXd forceMat;
Octree* octree = NULL;

// gravitational constant
const double G = 1;
// theta is a threshold value used in Barnes-Hut to determine which center-of-mass to use. The closer theta is to 0, the more accurate the simulation is (with diminishing returns).
const double THETA = 0.75;
// limiting the FPS to prevent strange looking accelerations
const double MAX_FPS = 60;

// This provides a bounding box around the entire simulation, if desired
double universeLength;
Eigen::Vector3d universePosition;

static void error_callback(int error, const char *description)
{
	std::cerr << description << std::endl;
}

static void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) 
	{
		glfwSetWindowShouldClose(window, GL_TRUE);
	}
}

static void char_callback(GLFWwindow *window, unsigned int key)
{
	keyToggles[key] = !keyToggles[key];
	switch (key) 
	{
	case ' ':
		paused = !paused;
		if (paused)
		{
			std::cout << "Pausing" << std::endl;
		}
		else
		{
			std::cout << "Playing" << std::endl;
		}
		break;
	case 'a':
		barnesHut = !barnesHut;
		if (barnesHut)
		{
			std::cout << "Running Barnes-Hut" << std::endl;
		}
		else
		{
			std::cout << "Running Naive" << std::endl;
		}
		break;
	case 'o':
		if (barnesHut)
		{
			drawOctree = !drawOctree;
			if (drawOctree)
			{
				std::cout << "Drawing Octree" << std::endl;
			}
			else
			{
				std::cout << "Disabling Octree drawing" << std::endl;
			}
		}
		break;
	case 'p':
		if (drawOctree)
		{
			drawEmptyLeaves = !drawEmptyLeaves;
			if (drawEmptyLeaves)
			{
				std::cout << "Drawing empty leaves" << std::endl;
			}
			else
			{
				std::cout << "Disabling empty leaf drawing" << std::endl;
			}
		}
		break;
	}
}

static void cursor_position_callback(GLFWwindow* window, double xmouse, double ymouse)
{
	int state = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
	if (state == GLFW_PRESS) 
	{
		camera->mouseMoved(xmouse, ymouse);
	}
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
	// Get the current mouse position.
	double xmouse, ymouse;
	glfwGetCursorPos(window, &xmouse, &ymouse);
	// Get current window size.
	int width, height;
	glfwGetWindowSize(window, &width, &height);
	if (action == GLFW_PRESS) 
	{
		bool shift = mods & GLFW_MOD_SHIFT;
		bool ctrl  = mods & GLFW_MOD_CONTROL;
		bool alt   = mods & GLFW_MOD_ALT;
		camera->mouseClicked(xmouse, ymouse, shift, ctrl, alt);
	}
}

static void initGL()
{
	GLSL::checkVersion();
	
	// Set background color
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	// Enable z-buffer test
	glEnable(GL_DEPTH_TEST);
	// Enable alpha blending
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	
	progSimple = make_shared<Program>();
	progSimple->setShaderNames(RESOURCE_DIR + "simple_vert.glsl", RESOURCE_DIR + "simple_frag.glsl");
	progSimple->setVerbose(false); // Set this to true when debugging.
	progSimple->init();
	progSimple->addUniform("P");
	progSimple->addUniform("MV");
	
	prog = make_shared<Program>();
	prog->setVerbose(true); // Set this to true when debugging.
	prog->setShaderNames(RESOURCE_DIR + "particle_vert.glsl", RESOURCE_DIR + "particle_frag.glsl");
	prog->init();
	prog->addUniform("P");
	prog->addUniform("MV");
	prog->addAttribute("aPos");
	prog->addAttribute("aTex");
	prog->addUniform("radius");
	prog->addUniform("alphaTexture");
	prog->addUniform("color");
	
	texture = make_shared<Texture>();
	texture->setFilename(RESOURCE_DIR + "alpha.jpg");
	texture->init();
	texture->setUnit(0);
	
	camera = make_shared<Camera>();
	
	// Initialize OpenGL for particles.
	for (auto p : particles) 
	{
		p->init();
	}
	
	GLSL::checkError(GET_FILE_LINE);
}

// Sort particles by their z values in camera space
class ParticleSorter 
{
public:
	bool operator()(size_t i0, size_t i1) const
	{
		// Particle positions in world space
		const Vector3d &x0 = particles[i0]->getPosition();
		const Vector3d &x1 = particles[i1]->getPosition();
		// Particle positions in camera space
		float z0 = V.row(2) * Vector4f(x0(0), x0(1), x0(2), 1.0f);
		float z1 = V.row(2) * Vector4f(x1(0), x1(1), x1(2), 1.0f);
		return z0 < z1;
	}
	
	void setViewMatrix(glm::mat4 V2)
	{
		for (int i = 0; i < 4; ++i) 
		{
			for(int j = 0; j < 4; ++j) 
			{
				V(i,j) = V2[j][i]; // indexing is different in Eigen and glm
			}
		}
	}
	
	Matrix4f V; // current view matrix
};
ParticleSorter sorter;

// http://stackoverflow.com/questions/1577475/c-sorting-and-keeping-track-of-indexes
template <typename T>
vector<size_t> sortIndices(const vector<T> &v) 
{
	// initialize original index locations
	vector<size_t> idx(v.size());
	for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;
	// sort indexes based on comparing values in v
	sort(idx.begin(), idx.end(), sorter);
	return idx;
}

void renderGL()
{
	// Get current frame buffer size.
	int width, height;
	glfwGetFramebufferSize(window, &width, &height);
	glViewport(0, 0, width, height);
	
	// Use the window size for camera.
	glfwGetWindowSize(window, &width, &height);
	camera->setAspect((float) width/(float) height);
	
	// Clear buffers
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	if (keyToggles[(unsigned) 'c']) 
	{
		glEnable(GL_CULL_FACE);
	} 
	else 
	{
		glDisable(GL_CULL_FACE);
	}

	if (keyToggles[(unsigned) 'l']) 
	{
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	} 
	else 
	{
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	}
	
	auto P = make_shared<MatrixStack>();
	auto MV = make_shared<MatrixStack>();
	
	// Apply camera transforms
	P->pushMatrix();
	camera->applyProjectionMatrix(P);
	MV->pushMatrix();
	camera->applyViewMatrix(MV);
	// Set view matrix for the sorter
	sorter.setViewMatrix(MV->topMatrix());
	
	// Draw particles
	prog->bind();
	texture->bind(prog->getUniform("alphaTexture"));
	glUniformMatrix4fv(prog->getUniform("P"), 1, GL_FALSE, glm::value_ptr(P->topMatrix()));
	// Sort particles by Z for transparency rendering.
	// Since we don't want to modify the contents of the vector, we compute the
	// sorted indices and traverse the particles in this sorted order.
	for (auto i : sortIndices(particles)) 
	{
		particles[i]->draw(prog, MV);
	}
	texture->unbind();
	prog->unbind();

	// Draw octree
	if (barnesHut && drawOctree)
	{
		progSimple->bind();
		glUniformMatrix4fv(progSimple->getUniform("P"), 1, GL_FALSE, glm::value_ptr(P->topMatrix()));
		glUniformMatrix4fv(progSimple->getUniform("MV"), 1, GL_FALSE, glm::value_ptr(MV->topMatrix()));
		octree->DrawLeaves(octree->getRoot(), drawEmptyLeaves);
		progSimple->unbind();
	}

	//////////////////////////////////////////////////////
	// Cleanup
	//////////////////////////////////////////////////////
	
	// Pop stacks
	MV->popMatrix();
	P->popMatrix();
	
	GLSL::checkError(GET_FILE_LINE);
}

void loadParticles(const char *filename)
{
	ifstream in;
	in.open(filename);
	if (!in.good()) 
	{
		std::cout << "Cannot read " << filename << std::endl;
		return;
	}

	// 1st line:
	// <n> <h> <e2>
	int n;
	in >> n;		// number of bodies
	in >> h;		// time step
	in >> e2;		// sigma squared, the softening length

	// Rest of the lines:
	// <mass> <position> <velocity> <color> <radius>
	float totMass = 0;

	double maxDimension = std::numeric_limits<double>::min();
	double minDimension = std::numeric_limits<double>::max();
	while (!in.eof())
	{
		float mass;
		Eigen::Vector3d pos;
		Eigen::Vector3d vel;
		Eigen::Vector3f color;
		float radius;

		in >> mass;
		in >> pos(0) >> pos(1) >> pos(2);
		in >> vel(0) >> vel(1) >> vel(2);
		in >> color(0) >> color(1) >> color(2);
		in >> radius;

		totMass += mass;

		auto particle = make_shared<Particle>();
		particle->setMass(mass);
		particle->setPosition(pos);
		particle->setVelocity(vel);
		particle->setColor(color);
		particle->setRadius(radius);

		double x = particle->getPosition()(0);
		if (x > maxDimension)
		{
			maxDimension = x + 1e-1;
		}
		if (x < minDimension)
		{
			minDimension = x - 1e-1;
		}

		double y = particle->getPosition()(1);
		if (y > maxDimension)
		{
			maxDimension = y + 1e-1;
		}
		if (y < minDimension)
		{
			minDimension = y - 1e-1;
		}

		double z = particle->getPosition()(2);
		if (z > maxDimension)
		{
			maxDimension = z + 1e-1;
		}
		if (z < minDimension)
		{
			minDimension = z - 1e-1;
		}

		particles.push_back(particle);
	}
	forceMat = Eigen::MatrixXd(3, particles.size());
	double length = maxDimension - minDimension;
	universeLength = length * 5;
	universePosition << minDimension * 5, minDimension * 5, minDimension * 5;

	in.close();
	std::cout << "Loaded galaxy from " << filename << std::endl;
	std::cout << "Universe mass: " << totMass << std::endl;
}

void stepParticlesNaive()
{
	for (int i = 0; i < particles.size(); i++)
	{
		auto p_i = particles[i];  
		Eigen::Vector3d force_i;
		force_i << 0, 0, 0;

		for (int j = 0; j < particles.size(); j++)
		{
			if (i == j)
			{
				continue;
			}

			auto p_j = particles[j];
			Eigen::Vector3d r = p_j->getPosition() - p_i->getPosition();
			double rNorm = r.norm();

			force_i += (G * p_i->getMass() * p_j->getMass()) / std::pow(rNorm * rNorm + e2, 1.5) * r;
		}

		forceMat.block<3, 1>(0, i) = force_i;
	}

	for (int i = 0; i < particles.size(); i++)
	{
		particles[i]->setVelocity(particles[i]->getVelocity() + (h/particles[i]->getMass())*forceMat.block<3, 1>(0, i));
		particles[i]->setPosition(particles[i]->getPosition() + h*particles[i]->getVelocity());
	}

	t += h;
}

void stepParticlesBarnesHut()
{
	octree = new Octree(particles);
	octree->ComputeAllCentersOfMass();
	octree->ComputeAllForces(particles, h, &forceMat, G, e2, THETA);

	//std::vector<std::shared_ptr<Particle>> aliveParticles = std::vector<std::shared_ptr<Particle>>();
	for (int i = 0; i < particles.size(); i++)
	{
		particles[i]->setVelocity(particles[i]->getVelocity() + (h/particles[i]->getMass())*forceMat.block<3, 1>(0, i));
		particles[i]->setPosition(particles[i]->getPosition() + h*particles[i]->getVelocity());

		/*if (IsPointInsideBoundingBox(particles[i]->getPosition(), universePosition, universeLength))
		{
			aliveParticles.push_back(particles[i]);
		}*/
	}
	//particles = aliveParticles;

	t += h;
}

int main(int argc, char **argv)
{
	if (argc < 2 || argc > 4)
	{
		std::cout << "Usage: BARNES-HUT <RESOURCE_DIR> <INPUT FILE>" << std::endl;
		std::cout << "   or: BARNES-HUT <#steps>       <INPUT FILE> <SIMULATION_ALGORITHM>" << std::endl;
		std::cout << "		 <SIMULATION_ALGORITHM> is `b` (barnes-hut) or `n` (naive)" << std::endl;
		exit(0);
	}

	// Create the particles
	loadParticles(argv[2]);

	// Try parsing `steps`
	int steps;
	int result = sscanf(argv[1], "%i", &steps);
	if (result)
	{
		// Run without OpenGL
		std::cout << "Running without OpenGL for " << steps << " steps" << std::endl;
		std::cout << "Number of particles: " << particles.size() << std::endl;
		
		bool barnesHutSimulation;
		if (strcmp(argv[3], "b") == 0 || strcmp(argv[3], "B") == 0)
		{
			barnesHutSimulation = true;
			std::cout << "Barnes-Hut simulation" << std::endl;
		}
		else if (strcmp(argv[3], "n") == 0 || strcmp(argv[3], "N") == 0)
		{
			barnesHutSimulation = false;
			std::cout << "Naive simulation" << std::endl;
		}
		else
		{
			barnesHutSimulation = true;
			std::cout << "Unrecognized <SIMULATION_ALGORITHM> defaulting to Barnes-Hut simulation" << std::endl;
		}
		
		MyTimer timer;
		timer.start();
		for (int k = 0; k < steps; ++k) 
		{
			if (barnesHutSimulation)
			{
				stepParticlesBarnesHut();
			}
			else
			{
				stepParticlesNaive();
			}
		}
		double elapsedMS = timer.elapsedMS();

		for (int k = 0; k < particles.size(); k++)
		{
			Particle p = *particles[k];
			//std::cout << p.getPosition() << std::endl << std::endl;
		}

		std::cout << "Simulation took " << elapsedMS << " milliseconds" << std::endl;
	} 
	else 
	{
		// `steps` could not be parsed
		std::cout << "Running with OpenGL" << std::endl;
		std::cout << "Space: pause\na: switch simulation algorithms\no: octree visualization" << std::endl;
		// Run with OpenGL until the window is closed
		RESOURCE_DIR = argv[1] + string("/");

		glfwSetErrorCallback(error_callback);
		if (!glfwInit()) 
		{
			return -1;
		}

		// Create a windowed mode window and its OpenGL context.
		window = glfwCreateWindow(1920, 1080, "Barnes-Hut Simulation", NULL, NULL);
		if (!window) 
		{
			glfwTerminate();
			return -1;
		}

		glfwMakeContextCurrent(window);
		glewExperimental = true;
		if (glewInit() != GLEW_OK) 
		{
			std::cerr << "Failed to initialize GLEW" << std::endl;
			return -1;
		}
		glGetError(); // A bug in glewInit() causes an error that we can safely ignore.
		std::cout << "OpenGL version: " << glGetString(GL_VERSION) << std::endl;
		std::cout << "GLSL version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;

		// Set vsync.
		glfwSwapInterval(1);
		glfwSetKeyCallback(window, key_callback);
		glfwSetCharCallback(window, char_callback);
		glfwSetCursorPosCallback(window, cursor_position_callback);
		glfwSetMouseButtonCallback(window, mouse_button_callback);
		initGL();

		// Loop until the user closes the window.
		double lastTime = glfwGetTime();
		while (!glfwWindowShouldClose(window)) 
		{
			// Step simulation.
			if (!paused)
			{
				if (barnesHut)
				{
					if (octree != NULL)
					{
						delete octree;
					}
					octree = NULL;
					stepParticlesBarnesHut();
				}
				else
				{
					stepParticlesNaive();
				}
				renderGL();
			}
			else
			{
				if (barnesHut)
				{
					if (octree != NULL)
					{
						renderGL();
					}
				}
				else
				{
					renderGL();
				}
			}
			glfwSwapBuffers(window);
			glfwPollEvents();

			// Capping the framerate of the simulation
			while (glfwGetTime() < lastTime + 1.0 / MAX_FPS)
			{
				// Sleep
			}
			lastTime += 1.0 / MAX_FPS;
		}


		glfwDestroyWindow(window);
		glfwTerminate();
		std::cout << "Elapsed time: " << (t * 3.261539827498732e6) << " years" << std::endl;
	}

	return 0;
}
