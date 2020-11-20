// Liam Bessell, 11/20/20

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
#include "Particle.h"
#include "Texture.h"

using namespace std;
using namespace Eigen;

bool keyToggles[256] = {false};				// only for English keyboards!

GLFWwindow *window;							// Main application window
string RESOURCE_DIR = "";					// Where the resources are loaded from

shared_ptr<Program> progSimple;
shared_ptr<Program> prog;
shared_ptr<Camera> camera;
vector< shared_ptr<Particle> > particles;
shared_ptr<Texture> texture;
double t, h, e2;

const double sigmaSquared = 1e-4;
const double BigG = 1.0;
Eigen::MatrixXd forceMat;

static void error_callback(int error, const char *description)
{
	cerr << description << endl;
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
	
	// If there were any OpenGL errors, this will print something.
	// You can intersperse this line in your code to find the exact location
	// of your OpenGL error.
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
		for(int i = 0; i < 4; ++i) {
			for(int j = 0; j < 4; ++j) {
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
	camera->setAspect((float)width/(float)height);
	
	// Clear buffers
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	if(keyToggles[(unsigned)'c']) {
		glEnable(GL_CULL_FACE);
	} else {
		glDisable(GL_CULL_FACE);
	}
	if(keyToggles[(unsigned)'l']) {
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	} else {
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
	for(auto i : sortIndices(particles)) {
		particles[i]->draw(prog, MV);
	}
	texture->unbind();
	prog->unbind();
	
	//////////////////////////////////////////////////////
	// Cleanup
	//////////////////////////////////////////////////////
	
	// Pop stacks
	MV->popMatrix();
	P->popMatrix();
	
	GLSL::checkError(GET_FILE_LINE);
}

void saveParticles(const char *filename)
{
	ofstream out(filename);
	if (!out.good()) 
	{
		std::cout << "Could not open " << filename << endl;
		return;
	}
	
	// 1st line:
	// <n> <h> <e2>
	out << particles.size() << " " << h << " " << " " << e2 << endl;

	// Rest of the lines:
	// <mass> <position> <velocity> <color> <radius>
	
	//
	// IMPLEMENT ME
	//
	
	out.close();
	std::cout << "Wrote galaxy to " << filename << endl;
}

void loadParticles(const char *filename)
{
	ifstream in;
	in.open(filename);
	if (!in.good()) 
	{
		std::cout << "Cannot read " << filename << endl;
		return;
	}

	// 1st line:
	// <n> <h> <e2>
	int n;
	in >> n;
	in >> h;
	in >> e2;

	// Rest of the lines:
	// <mass> <position> <velocity> <color> <radius>
	
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

		auto particle = make_shared<Particle>();
		particle->setMass(mass);
		particle->setPosition(pos);
		particle->setVelocity(vel);
		particle->setColor(color);
		particle->setRadius(radius);

		particles.push_back(particle);
	}
	forceMat = Eigen::MatrixXd(3, particles.size());

	in.close();
	std::cout << "Loaded galaxy from " << filename << endl;
}

void createParticles()
{
	srand(0);
	t = 0.0;
	h = 1e-2;
	e2 = 1e-4;
	double r = 1.0;							// distance between stars
	double a = 1.0;							// length of semi-major axis

	Eigen::Vector3d pos;
	Eigen::Vector3d vel;

	auto heavyStar = make_shared<Particle>();
	pos << 0, 0, 0;
	vel << 0, 0, 0;
	heavyStar->setMass(1e-2);
	heavyStar->setPosition(pos);
	heavyStar->setVelocity(vel);
	particles.push_back(heavyStar);

	auto lightStar = make_shared<Particle>();
	pos << r, 0, 0;
	vel << 0, std::sqrtf(BigG * heavyStar->getMass() * (2/r - 1/a)), 0;
	lightStar->setMass(1e-6);
	lightStar->setPosition(pos);
	lightStar->setVelocity(vel);
	particles.push_back(lightStar);

	forceMat = Eigen::MatrixXd(3, particles.size());
}

void stepParticles()
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
			Eigen::Vector3d position_ij;
			position_ij = p_j->getPosition() - p_i->getPosition();
			float distance_ij = position_ij.norm();

			force_i += ((BigG * p_i->getMass() * p_j->getMass()) / std::powf(distance_ij * distance_ij + sigmaSquared, 1.5)) * position_ij;
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

int main(int argc, char **argv)
{
	if (argc != 2 && argc != 3)
	{
		// Wrong number of arguments
		std::cout << "Usage: Lab18 <RESOURCE_DIR> <(OPTIONAL) INPUT FILE>" << endl;
		std::cout << "   or: Lab18 <#steps>       <(OPTIONAL) INPUT FILE>" << endl;
		exit(0);
	}
	// Create the particles...
	if (argc == 2) 
	{
		// ... without input file
		createParticles();
	} 
	else 
	{
		// ... with input file
		loadParticles(argv[2]);
	}
	// Try parsing `steps`
	int steps;
	if (sscanf(argv[1], "%i", &steps)) 
	{
		// Success!
		std::cout << "Running without OpenGL for " << steps << " steps" << endl;
		// Run without OpenGL
		for (int k = 0; k < steps; ++k) 
		{
			stepParticles();
		}

		for (int k = 0; k < particles.size(); k++)
		{
			Particle p = *particles[k];
			std::cout << p.getPosition() << std::endl << std::endl;
		}
	} 
	else 
	{
		// `steps` could not be parsed
		std::cout << "Running with OpenGL" << endl;
		// Run with OpenGL until the window is closed
		RESOURCE_DIR = argv[1] + string("/");
		// Set error callback.
		glfwSetErrorCallback(error_callback);
		// Initialize the library.
		if (!glfwInit()) 
		{
			return -1;
		}
		// Create a windowed mode window and its OpenGL context.
		window = glfwCreateWindow(1280, 960, "Barnes-Hut Simulation", NULL, NULL);
		if (!window) 
		{
			glfwTerminate();
			return -1;
		}
		// Make the window's context current.
		glfwMakeContextCurrent(window);
		// Initialize GLEW.
		glewExperimental = true;
		if (glewInit() != GLEW_OK) 
		{
			cerr << "Failed to initialize GLEW" << endl;
			return -1;
		}
		glGetError(); // A bug in glewInit() causes an error that we can safely ignore.
		std::cout << "OpenGL version: " << glGetString(GL_VERSION) << endl;
		std::cout << "GLSL version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << endl;
		// Set vsync.
		glfwSwapInterval(1);
		// Set keyboard callback.
		glfwSetKeyCallback(window, key_callback);
		// Set char callback.
		glfwSetCharCallback(window, char_callback);
		// Set cursor position callback.
		glfwSetCursorPosCallback(window, cursor_position_callback);
		// Set mouse button callback.
		glfwSetMouseButtonCallback(window, mouse_button_callback);
		// Initialize scene.
		initGL();
		// Loop until the user closes the window.
		while (!glfwWindowShouldClose(window)) 
		{
			// Step simulation.
			stepParticles();
			// Render scene.
			renderGL();
			// Swap front and back buffers.
			glfwSwapBuffers(window);
			// Poll for and process events.
			glfwPollEvents();
		}
		// Quit program.
		glfwDestroyWindow(window);
		glfwTerminate();
	}
	std::cout << "Elapsed time: " << (t*3.261539827498732e6) << " years" << endl;
	return 0;
}
