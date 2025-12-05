## Loading and Editing of Scenes
# Use the load_scene() function to load a scene with and without merging objects
# Learn how to add and remove objects from a scene
# Learn how to translate, rotate, and scale objects within a scene

import drjit as dr
import mitsuba as mi

# Import or install Sionna
try:
    import sionna.rt
except ImportError as e:
    import os
    os.system("pip install sionna-rt")
    import sionna.rt

no_preview = True # Toggle to False to use the preview widget
                  # instead of rendering for scene visualization

from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, Camera,\
                      PathSolver, ITURadioMaterial, SceneObject

## Loading Scenes and Merging Objects
scene = load_scene(sionna.rt.scene.simple_street_canyon,
                   merge_shapes=False) # Disable merging of objects

for name, obj in scene.objects.items():
    print(f'{name:<15}{obj.radio_material.name}')

# Let's now reload the scene with the merging of objects enabled:
scene = load_scene(sionna.rt.scene.simple_street_canyon,
                   merge_shapes=True) # Enable merging of objects (default)

for name, obj in scene.objects.items():
    print(f'{name:<15}{obj.radio_material.name}')

# Let's exclude buildings with indices smaller than 3 from the merging process:
scene = load_scene(sionna.rt.scene.simple_street_canyon,
                   merge_shapes=True, # Enable merging of objects
                   merge_shapes_exclude_regex=r'building_[0-2]$') # Exclude from merging
                                                                  # buildings with indices < 3

for name, obj in scene.objects.items():
    print(f'{name:<15}{obj.radio_material.name}')

# We can see that “building_1” and “building_2” have not been merged. 
# As a result, “building_5” has not been merged either, as it has no other objects to be merged with.

## Editing Scenes
# Let's load a more complex scene and visualize it
scene = load_scene(sionna.rt.scene.etoile) # Objects are merged by default


cam = Camera(position=[-360,145,400], look_at=[-115,33,1.5])
if no_preview:
    scene.render(camera=cam);
else:
    scene.preview();

# Next, we will add a few objects to the scene.
# We will add cars made of metal to the scene.

# Number of cars to add
num_cars = 20

# Radio material constituing the cars
# We use ITU metal, and use red color for visualization to
# make the cars easily discernible
car_material = ITURadioMaterial("car-material",
                                "metal",
                                thickness=0.01,
                                color=(0.8, 0.1, 0.1))

# Instantiate `num_cars` cars sharing the same mesh and material
cars = [SceneObject(fname=sionna.rt.scene.low_poly_car, # Simple mesh of a car
                    name=f"car-{i}",
                    radio_material=car_material)
        for i in range(num_cars)]

# Add the list of newly instantiated objects to the scene
scene.edit(add=cars)

if no_preview:
    scene.render(camera=cam);
else:
    scene.preview();

# We can see the red cars in the scene, but because they are all located at the same position,
# it appears that only a single car was added to the scene.

# In the next, we will position the cars in the scene and also set their orientations.

# Positions
# Car are positioned in a circle around the central monument
# Center of the circle
c = mi.Point3f(-127, 37, 1.5)
# Radius of the circle
r = 100
# Angles at which cars are positioned
thetas = dr.linspace(mi.Float, 0., dr.two_pi, num_cars, endpoint=False)
# Cars positions
cars_positions = c + mi.Point3f(dr.cos(thetas), dr.sin(thetas), 0.)*r

# Orientations
# Compute points the car "look-at" to set their orientation
d = dr.normalize(cars_positions - c)
# Tangent vector to the circle at the car position
look_at_dirs = mi.Vector3f(d.y, -d.x, 0.)
look_at_points = cars_positions + look_at_dirs

# Set the cars positions and orientations
for i in range(num_cars):
    cars[i].position = mi.Point3f(
        cars_positions.x[i], 
        cars_positions.y[i], 
        cars_positions.z[i])
    cars[i].look_at(mi.Point3f(
        look_at_points.x[i], 
        look_at_points.y[i], 
        look_at_points.z[i]))

if no_preview:
    scene.render(camera=cam);
else:
    scene.preview();

# Objects can also be scaled. This is useful, for example, when the scale of the mesh from which the object is built does not suit the scene.
# To illustrate this feature, let's scale the first car to be twice as large as the other cars.

cars[0].scaling = 3.0

if no_preview:
    scene.render(camera=cam);
else:
    scene.preview();

# Finally, objects can be removed from the scene using the Scene.edit() function
# To illustrate this, let's remove the last car we have added.

scene.edit(remove=[cars[-1]])

if no_preview:
    scene.render(camera=cam);
else:
    scene.preview();


## Path Computation with the Edited Scene

# Let's compute radio propagation paths on the edited scene.
# We start by adding a transmitter on the roof of an arbitrarily selected building,
# as well as a receiver on top of each car.
# We also set the transmitter and receiver arrays.


# Add a transmitter on top of a building
scene.remove("tx")
scene.add(Transmitter("tx", position=[-36.59, -65.02, 25.], display_radius=2))

# Add a receiver on top of each car
for i in range(num_cars):
    scene.remove(f"rx-{i}")
    scene.add(Receiver(f"rx-{i}", position=[cars_positions.x[i],
                                            cars_positions.y[i],
                                            cars_positions.z[i] + 3],
                      display_radius=2))


# Set the transmit and receive antenna arrays
scene.tx_array = PlanarArray(num_cols=1,
                             num_rows=1,
                             pattern="iso",
                             polarization="V")
scene.rx_array = scene.tx_array

# We are now ready to compute paths

p_solver = PathSolver()
paths = p_solver(scene, max_depth=5)

if no_preview:
    scene.render(camera=cam, paths=paths);
else:
    scene.preview(paths=paths);

