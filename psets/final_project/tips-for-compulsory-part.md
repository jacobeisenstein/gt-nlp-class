## Some pointers for the compulsory part of the project ##

Below are some pointers for you to get started with the compulsory part of the project. 
-- Umashanthi

In order to construct the POS tag and headword set features you will need to integrate your external tagging/parsing results to the existing model. One way of doing this is to modify the SpanNode datastructure and add attributes for the POS tag and heardword set for each node.  Then you can assign values to these attributes while reading the dis/edu files.  

When you do the training, as you would have seen, the buildtree() method in the buildtreey.py reads the text (from the dis files) and creates nodes by calling the createnode() method. You can modify these methods to assign values for the new attributes you added to the datastructure. For the purpose of creating a mapping between the dis/edu files and your parsing output, you can pass a filename identifier when you are reading the training/test file instances.  You will also need to modify the backprop() method to incorporate the new attributes of the SpanNode when merging two nodes together.  You can see how the "text" attribute is constructed for the merged nodes and follow a similar approach for your new attributes.

When you do RST parsing on the test data (without evaluation), the init() method in the parser.py is the place whether the nodes get created. So you should also modify this method to assign values to the new SpanNode attributes. The operate() method in the parser.py is where the parsing action is performed and you need to merge attributes from multiple nodes when doing the reduce operation.  Again you can use the assignment method for the "text" attribute as an example. 

If you have modified the code to reflect the new SpanNode attributes, then you can directly access them in your feature.py when constructing the features. There might be mismatches when you map your tagging/parsing output to the dis/edu text units. So you should try to reduce/eliminate the mismatches as much as possible by adding some check points.

**Note:** The method explained above is just one way of integrating the tagging/parsing output to the source code provided for the project. So feel free to use your own approach. 