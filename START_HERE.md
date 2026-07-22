# README ABOUT THE PYTHON TEMPLATE.

This describes how to use the python template to create a new project,
rather than providing a preliminary README for the new project.

The following steps must be done manually for now. They could be automated
at some point.

1. Determine your project's name and then
    a. Copy this `python_template` folder to a new folder, and rename it to your project's name.
    b. Change `my_project` to that name thoughout, namely,
    * in the subfolder of `src/` 
    * in the README
    * in the Makefile, at the `NAME` variable 
    * in setup.py, at the `name` and `package_data` fields of the call to the `setup` function.  (Consider also updating the `description` field.)
    c. Remove START_HERE.md 

2. Link your new project to version control (assumed to be github):
    a. Create a github project with your project's name.
    b. Set or change the local project to have the correct remote url: https://docs.github.com/en/github/using-git/changing-a-remotes-url
    c. In setup.py, Consider updating the `url` field of the call to the `setup` function to match the github url.


# ACKNOWLEDGEMENTS

This template borrows heavily from work by Dan Lidral Porter and Tim Hopper, both of whom were gifted Data Science Engineers at Cylance, Inc. 


