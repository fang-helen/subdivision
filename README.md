## Subdivision algorithms

This is the final project I did for my computer graphics class at UT Austin (CS354), which I chose to do on polymesh subdivision algorithms.

This project was completely independent and self-scoped - we had discussed a couple of these algorithms in class but had not gone into the implementation details.

This is an OpenGL project that can be built using cmake.

Most of the core logic lives in in `subdivision/src/subdivision.cc`, and I briefly discuss it in the project writeup, which also includes screenshot demos.

### Algorithms

I implemented the Catmull clark, Loop, and Doo-sabin algorithms.

### Inputs

I also implemented OBJ file loading to render and subdivide your own polymeshes. Some of the OBJ files I used to to test my program are included in the obj/ folder.

### Running the project

I've included a runnable exe in the `exe/` folder. It can be run from the commmand line to specity the OBJ file to load, e.g. `> menger.exe ..\obj\teapot.obj`.

Because this project was built on top of one of our other class projects, some necessary files have been ommitted from this repo. I can provide the full project upon request, which can be built using cmake.
