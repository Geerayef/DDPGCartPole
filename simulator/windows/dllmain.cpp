/*
    Build this source code to cartpole.dll and place it in the same folder with cartpole.exe.
    Get the simulator at https://ailab.si/domen/cartpole/
*/

#include <Windows.h>
#include <string>
#include "engine.h"

#define DLLEXPORT extern "C" __declspec(dllexport)

BOOL APIENTRY DllMain(
    HINSTANCE hinstDLL,
    DWORD fdwReason,
    LPVOID lpReserved)
{
    switch (fdwReason) {
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
    case DLL_PROCESS_DETACH:
        break;
    }
    return TRUE;
}

std::string slog;
const char* plog = NULL;
#define LOG(s) { slog += s; plog = slog.c_str(); }

bool active = false;

DLLEXPORT void simulatorInitialize(SimulatorParameters& simulatorParameters)
{
    /* Called when the simulator is starting up. */
    /* Modify individual variables or keep their default values. */
    simulatorParameters.engineName = "Example engine";
    simulatorParameters.width = 1350;
    simulatorParameters.height = 800;
    simulatorParameters.log = &plog;
    simulatorParameters.camera.x = -6;
    simulatorParameters.camera.zoom = 0.9;

    /* Add craters or hills. */
    static const Crater craters[] = {
        { -6, 10, 1.5 }, // { position, width, depth }
        { 0, 0, 0 }      // Array must end with a zero-crater.
    };
    simulatorParameters.craters = craters;

    /* Add vertical markers. */
    static const Marker markers[] = {
        { 0, 0.1, 192, 192, 64 }, // { position, width, red, green, blue }
        { 0, 0, 0, 0, 0 }         // Array must end with a zero-marker.
    };
    simulatorParameters.markers = markers;

    LOG("Engine initialization successful.\n");
}

DLLEXPORT void simulatorShutdown()
{
    /* Called when the simulator is shutting down. */
    LOG("Simulator is shutting down.\n");
}

DLLEXPORT void setInitialState(SimulationState& simulationState)
{
    /* Called when the simulation is being reset. Keep the default initial state or modify individual values. */
    LOG("Simulation has been reset.\n");
    simulationState.x = -12; // The default is x = 0.
}

DLLEXPORT int stateUpdated(double simulationTime, SimulationState simulationState)
{
    /* Called when the simulator state has changed. */
    /* Return 1 to reset the simulation (e.g., invalid/fail state). */
    return 0;
}

DLLEXPORT void applyAction(CartAction& cartAction)
{
    /* Called when the simulator wants the engine to execute an action. */
    /* The default is zero-action. */
    if (active) {
        cartAction.direction = ActionDirection::RIGHT;
        cartAction.force = 1.0;
        cartAction.options = APPLY_DIRECTION_AND_FORCE;
    }
}

DLLEXPORT int keyPressed(KeyInfo& keyInfo)
{
    /* Called when the user presses/releases a key, and before the simulator responds to it.  */

    /* We turn the engine execution on/off with the space key. */
    if (keyInfo.state == KeyState::PRESSED) {
        if (keyInfo.code == 0x20) { // Space bar
            active = !active;
            LOG("Execution is ");
            LOG((active ? "ON.\n" : "OFF.\n"));

            /* Return a non-zero value to prevent the simulator to respond to the pressed key. */
            return 1;
        }
    }

    /* Return 0 to let the simulator process and respond to the pressed key. */
    return 0;
}