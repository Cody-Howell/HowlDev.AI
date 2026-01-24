# HowlDev.AI

This contains a Core project, which is what your game or logic calls for interfaces. Right now, there's only one that interfaces with the Genetic algorithm, so my Asteroids project will take in the Core and implement the `IGeneticRunner` (names are likely to change in the future). 

Then another project will get the Asteroids game and the Training class (which will pull in the other two AI projects) and run the genetic algorithm, with the options provided in the constructor. 

This is my first attempt at training a network, many things aren't very well tested, and I don't have a local project I can use to see if my training algo throws any errors at all. So I may list multiple versions with something like "Fixing training algorithm" and just have a range instead. 

0.1.1 (1/23/26)

- Set FunctionActivation as a parameter in the algorithm. 

0.1.0 (1/23/26)

- Making another attempt. I adjusted how the data comes out (I give a class instead of just, "Here's a file to write"), and we'll see if that works better. 
- Added Parallel.For instead of awaiting tasks, I hope to see a performance improvement. 

0.0.5 (1/4/26)

- Got it working!
- Added in a new Saving option for what networks you want to save; whether that be All, Survivors for each generation, or just the Best (whether or not they got culled by the end of the generation)

0.0.1 - 0.0.4 (1/3/26 - 1/4/26)

- Initialization and testing the genetic algorithm
    - Added in saving of the networks, since training is pointless if you can't get the network at the end...
- (and later added the correct wiki link)
