# droneSim

General modular Python simulator for quadrotors

# General

This is a simple quadrotor (drone) simulator with four basic elements. Though, this simulator can also work with other systems other than the drone. 
- The `model` block hosts the main model of the system. It should output the forces and moments of the system.
- The `controller` block hosts the controller(s) applied to the system. This can be a collection of different controllers, so long as they output a single control action (e.g. combined control action). 
- The `actuator` block describes the actuator dynamics, and takes the control action as an input, and outputs the 'true' action. 
- The `EOM` block gives the equations of motion. 

Finally, everything interacts through the `sim.py` file. This acts as the simulator, and hosts the updated loops. 


# Example

Provided is an example simulation of a mass-spring-damper system.

Users can run this simulation through the `exampleSim.py` (N.B NOT `sim.py`) and experiment with the initial conditions and reference signal.



# Usage

For convenience, users can run the `buildMySim.py` file to create your simulation files, with the necessary I/O streams of each module. In essence, this will construct all the necessary files (e.g. controller, model, actuator etc.) and link them all in a simulation file which you will use to run your simulation. 

When run, users are asked to provide a name for the simulation, for example `mySim`. 
The following files are then created (if the default option is selected)
- `mySim_actuator.py` located in the actuators folder. Here, users may specify their actuator dynamics using the provided template. For drone simulations, there is already a simple actuator model, `droneRotors.py`, which describes the rotor dynamics provided. 
- `mySim_controller.py` located in the controllers folder. Here, users can construct their controller(s). There is already a `PID` class available for import to easily construct PID controller(s). See `dronePIDController.py` for how to initialize the PID controllers.
- `mySim_EOM.py` located in the funcs folder. Here, users define the equations of motion pertaining to their case. Again, for drone simulations, the standard EOM are provided in `droneEOM.py`.
- `mySim_Model.py` located in the models folder. Here, users define/import/load their model. Currently, there are a few polynomial quadrotor models available identified from real flight data. Each model is valid across a different speed domain.
- `mySim.py` located in the main directory. This is where users interact with the simulation by defining parameters for the simulation blocks (e.g. rate constants for actuators), initial conditions, reference signals and so on. 

Using the `buildMySim.py` file will automatically link and import all the simulation blocks in the created `mySim.py` file. Users need only modify the blocks (scripts) they require. 
