# HowlDev.AI

This contains a Core project, which is what your game or logic calls for interfaces. Right now, there's only one that interfaces with the Genetic algorithm, so my Asteroids project will take in the Core and implement the `IGeneticRunner` (names are likely to change in the future). 

Then another project will get the Asteroids game and the Training class (which will pull in the other two AI projects) and run the genetic algorithm, with the options provided in the constructor. 

This is my first attempt at training a network, many things aren't very well tested, and I don't have a local project I can use to see if my training algo throws any errors at all. So I may list multiple versions with something like "Fixing training algorithm" and just have a range instead. 

0.0.1 - ??? (1/3/26 - ???)

- Initialization and testing the genetic algorithm
    - Added in saving of the networks, since training is pointless if you can't get the network at the end...
- (and later added the correct wiki link)
