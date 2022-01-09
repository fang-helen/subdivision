## Subdivision algorithms

This is the final project I did for my computer graphics class at UT Austin (CS354), which I chose to do on polymesh subdivision algorithms.

This project was completely independent and self-scoped - we had discussed a couple of these algorithms in class but had not gone into the implementation details.

This is an OpenGL project that can be built using cmake.

Most of the core logic lives in in [`subdivision/src/subdivision.cc`](https://github.com/fang-helen/subdivision/blob/master/subdivision/src/subdivision.cc), and I also briefly discuss it in the [project writeup](https://github.com/fang-helen/subdivision/blob/master/subdivision/project_writeup.pdf), which also includes screenshot demos.

### Algorithms

I implemented the Catmull clark, Loop, and Doo-sabin algorithms.

### Inputs

I also implemented OBJ file loading to render and subdivide your own polymeshes. Some of the OBJ files I used to to test my program are included in the [`obj/`](https://github.com/fang-helen/subdivision/tree/master/obj) folder.

### Controls
- J - perform 1 round of subdivision using the currently selected subdivision algorithm.
(The startup message in the terminal window will tell you which one).
- K - re-initializes the mesh using either the OBJ file or default cube, allowing you to start
the subdivision over.
- L - cycles through the subdivision algorithms so you can switch which method the program uses.

Camera controls:
- W/S - zoom in and out
- A/D, up/down - move camera around
- right/left - spin camera
- left mouse click + drag - another way to rotate the camera
- C - toggle orbital/FPS mode on camera

### Running the project

I've included a runnable exe in the [`exe/`](https://github.com/fang-helen/subdivision/tree/master/exe) folder. Run the binary from the command line to pass OBJ filename arguments, e.g. `> menger.exe ..\obj\teapot.obj`. If no argument is included, a cube will be initialized.

I can provide the full project upon request, which can be built using cmake. Because this project was built on top of one of our other class projects, some necessary files are currently ommitted from this repo. 
