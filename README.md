# HowlDev.Simulation.Physics

## HowlDev.Simulation.Physics.Primitive2D

Contains a few 2D-plane generics for simple calculations (it uses doubles, sometimes rounded). 

### Rotation2D

Holds an angle between 0 and 360 (with a precision of 2 digits). Has an extreme number of helper methods to point at given points, get distances between other angles, and to flip the angle. 

### Point2D

Holds two full double-precision points for X and Y. Has a number of helper methods to get distances in 2D space and midpoints, as well as some fancy operator functions. 

### Vector2D

Holds a Rotation2D object and a double for the velocity. Allows you to calculate points and angles off of others using operator functions. 

### Equation2D

Holds two coefficient values (for the slope and the y-intercept). Does most of the calculating for the Line2D. Not really intended to be used directly. 

### Line2D

Holds two Point2D objects in an array. Has logic for intersecting lines and retrieving intersecting points. 

### Circle2D

Holding a center Point2D and a radius, has methods for intersection with other circles and points. 

Read the wiki (which has links to the package) and get a bunch of class definitions! [Those exist here](https://wiki.codyhowell.dev/2dphysicslibrary).

# Changelog

0.8.2 (12/19/25)

- Vector2D has some additional operators that I hope will be useful in making my game easier. 
    - *, /, and the unary - operator (to flip it immediately). 
- Point2D now has as vector-getter from one point to another. (this -> parameter)

0.8.1 (12/19/25)

- Circle2D now has methods to change one property and leave the rest.

0.8 (12/19/25)

- BREAKING CHANGE
    - Changed Rotation, Point, Vector, Line, and Equation to Value types (structs) because I was foolish and it's what I should have done in the first place. 
        - I do kinda dislike how the API turned out, but I kept all the methods I wanted to, so...
- Created Circle2D struct with minimal methods, I'll see how many more I need to add. And I just want to get it into production to start testing it, so I'm going to publish. 

0.7.5 (12/17/25)

Somehow I don't have a changelog for everything I've done up to this point. Well, I have a bunch of classes, and I'm just working through all the primitives I want to contain. Right now, I'm building it so I can make a simple Asteroids clone, so my work will focus on the primitives I need for that to work. 

Also, testing the workflow. 
