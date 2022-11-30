#pragma once

enum ActionDirection {
	STOP = 0,
	RIGHT = 1,
	LEFT = 2
};

enum ActionOptions {
	APPLY_NO_ACTION = 0,
	APPLY_DIRECTION = 1,
	APPLY_DIRECTION_AND_FORCE = 2
};

enum KeyState {
	RELEASED = 0,
	PRESSED = 1
};

struct ObjectParameters {
	double size;
	double mass;
	double damping;
};

struct CameraParameters {
	double x;
	double y;
	double zoom;
};

struct Marker {
	double x;
	double width;
	unsigned char red;
	unsigned char green;
	unsigned char blue;
};

struct Crater {
	double x;
	double width;
	double depth;
};

struct SimulatorParameters {
	const char* engineName;
	int width;
	int height;
	int frequency;
	double gravity;
	double force;
	const Marker* markers;
	const Crater* craters;
	const char** log;
	ObjectParameters cart;
	ObjectParameters pole;
	CameraParameters camera;
};

struct InitialState {
	double x;
	double dx;
	double ddx;
	double theta;
	double dtheta;
	double ddtheta;
};

struct SimulationState {
	double x;
	double y;
	double phi;
	double dx;
	double ddx;
	double theta;
	double dtheta;
	double ddtheta;
};

struct CartAction {
	ActionDirection direction;
	double force;
	ActionOptions options;
};

struct KeyInfo {
	unsigned int code;
	char character;
	KeyState state;
};