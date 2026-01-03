# Graphs are All You Need: Generating Multimodal Representations for VQA

**Visual Question Answering requires understanding and relating text and image inputs. Here we use Graph Neural Networks to reason over both input modalities and improve performance on a VQA dataset.**

*Rajas Bansal · 10 min read · Jan 12, 2022*

---

By Dhruva Bansal, Drew Kaul, Rajas Bansal as part of the Stanford CS224W course project. In this blog post, we apply Graph ML to improve performance for VQA. We use a subset of the CLEVR dataset for our experiments. We explain and implement the paper “Multimodal Graph Networks for Compositional Generalization in Visual Question Answering” by Raeid Saqur et al. published at NeurIPS 2020 [1].


*A set of question-image pairs and their answers [2]*

Given the above question “What is the sports position of the man in the orange shirt?” and the image of two soccer players, how can we get a model to answer “goalie”? It’s a seemingly simple task for us as human beings — with a glance at the image, we almost instantly recognize the player in orange as a goalie. However, answering this question requires a complex understanding of both images and language — the ability to identify different objects in a scene, parse and understand parts of a question, and relate visual and linguistic concepts.

In this post, we’ll explore Visual Question Answering (VQA), a task in which our model receives an image $im$ and a text-based question $t$ and outputs the answer to the question. In the process, we’ll learn how to use graph ML techniques to perform multimodal learning — learning joint representations of multiple data modalities — and implement them ourselves with PyG.

## Why Graph ML for VQA

In order to solve the VQA task, we need to develop a model that not only understands text and images but also the interactions between their concepts. A natural way to capture the relations between visual and linguistic concepts is to represent them as graphs — an image is a set of related objects with particular attributes, and text is a set of concepts linked together by relationships.

By constructing graphs from our inputs, we can leverage techniques in graph machine learning to solve VQA. There are two main advantages to this approach. First, parsing images and text into graphs helps us learn explicit correspondences between visual objects and linguistic concepts at a more fine grained level. Second, our representations can scale better to handle longer range dependencies in our text, since graph neural networks can handle multi-hop reasoning between nodes.


*The overall architecture of MGN [1]*

## The Approach

We will use the Multimodal Graph Network (MGN) from [1] in order to learn a multimodal representation $h_{im, t}$ of our image-question pair and then use $h_{im, t}$ to generate an answer. In the rest of this post, we will walk through the following steps:

1.  Processing our input question into a graph $G_t$ and image into a graph $G_{im}$ using the Graph Parser.
2.  Passing the text graph $G_t$ and image graph $G_{im}$ into a graph neural network (GNN) to get the text and image node embeddings.
3.  Combining the embeddings using the Graph Matcher, which projects the text embeddings into the image embedding space and returns the combined multimodal representation of the input.
4.  Passing the joint representation through a sequence to sequence model to output the answer to the question.

## Parsing the image and text into graphs

Let’s consider the multimodal input for VQA to be the tuple $(im, t)$, where $im$ is our image and $t$ is the text containing the question we’d like to answer about our image. We first need to parse our input text and image into graphs, where the nodes represent objects and the edges represent the relationships between these concepts. To do so, we will use the Graph Parser developed in [1].

Specifically, the Graph Parser will take $(im, t)$ as input and return the object graphs $G_{im} = (V_{im}, A_{im}, X_{im}, E_{im})$ and $G_t = (V_t, A_t, X_t, E_t)$. Here, $V$ is the set of nodes, $A$ is the adjacency matrix, $X \in R^{|V| \times D}$ is the node feature matrix, and $E \in R^{|E| \times D}$ is the edge feature matrix.


*The architecture of Mask-RCNN, which is used to parse our images [3]*

For the image $im$, the Graph Parser uses a Mask-RCNN and a Resnet-50 FPN to extract the objects and attributes to construct the nodes and edges of $G_{im}$. For the text $t$, the parser uses an entity recognizer to extract objects and a relation matcher to extract the edges of $G_t$. The matrices $X$ and $E$ are then obtained by using a language model on the extracted object and attribute labels as well as relations to get the initial feature embeddings.

## Solving VQA with Graph ML

Now that we have our image scene graph $G_{im}$ and text graph $G_t$, we can apply graph neural networks to update our image and text node embeddings. We will later combine these embeddings to generate a single multimodal representation of both graphs. In the next two sections, we provide some background on GNNs as well as GIN, the specific network which we use here. In the third section, we discuss how to apply GIN to generate our node embeddings.


*The graph neural network framework: message, aggregate, and merge [4]*

### Graph Neural Networks (GNN)

GNNs are neural networks that allow us to process graphs of arbitrary sizes. To ensure that the ordering of our nodes doesn’t affect the output, GNNs contain permutation invariant operations. We can use GNNs for a variety of tasks, including learning representations for individual nodes, edges, subgraphs, and even entire graphs! These representations can then be used to solve downstream tasks such as classification, or as in our case, visual question answering.

Graph neural networks compute node representations by performing a message passing algorithm. Each node constructs a message from its embedding, aggregates the messages of all its neighbors, and then updates itself using its own message and the aggregated messages of its neighbors. After $k$ GNN layers, the node $v$ has an updated representation $h_v^{(k)}$, which summarizes information from the node’s $k$-hop neighborhood.

### Graph Isomorphism Network (GIN)



The choice of aggregate and merge functions in GNNs is crucial. Different choices for the aggregate and merge functions can lead to different variants of GNN. For example, the sum function for aggregate leads to the Graph Convolutional Network (GCN) while using a concatenation function for the merge function leads to GraphSAGE.


*The sum, mean, and max aggregators over a multi-set, ranked by expressive power [5]*

The representational power of the GNN depends on the aggregate function used [5]. The image above shows the ranking of the expressive power of GNNs based on their aggregation function. A GNN variant with maximal representation power would be one which has an injective aggregate function, like the Graph Isomorphism Network. The combination of the merge and the aggregate equation for GIN is shown below:



This is the node update equation for GIN. The aggregate function used here is injective to maximize the expressiveness of the GNN.

### Applying GIN to produce the image and text node embeddings

Now, let’s apply GIN to update the node embeddings in our image and text graphs. We can use PyG in order to implement the GIN layer. In the code snippet below, we walk through an implementation of multi-layer GIN:

Here, the GINConv layer takes the input features of the nodes and graph structure and outputs the updated node features for each node. The output node features from each layer are concatenated and passed through a linear layer to get the final node features.

## Constructing a multimodal representation using Graph Matcher

Using GIN, we got the image and text node embeddings for both graphs. Now, we can use the Graph Matcher from [1] to learn correspondences between the image and text nodes. These correspondences help us learn how the image and the text nodes relate to each other, and aid in constructing a multimodal representation that considers these interactions. Since the image and the text nodes are embedded in different embedding spaces, the graph matcher uses the correspondence matrix to project the text embeddings into the image embedding space, which allows for their concatenation (as they would have similar meanings).

Let  be the graph constructed from the image input and  be the graph constructed from the text input. Running a GIN on both of these graphs separately gives us embedding matrices  and , where the -th row is the embedding of vertex .
Using  and  , we get a soft correspondence matrix,  between the nodes  and  by

where the -th row vector  is a probability distribution over potential correspondences to nodes in  for .
Intuitively, every element of this matrix can be thought of a “likelihood score” to measure how well the two nodes would match together. In order to convert this likelihood score into a probability distribution, we apply the softmax activation to this matrix.
Using , we can project a text embedding into the image embedding space:

Thus, using  we project the whole  to get a new matrix  which is the projection of the text graph into the image embedding space. As both of these matrices are now in the same embedding space, we can concatenate the two, to get .

In order to get one vector to summarize the whole graph, we can use a permutation invariant aggregation function like mean to get a single vector . Thus, we now have the combined multimodal representation of the image and text graphs .
Now, let’s walk through an example implementation of the Graph Matcher.



In lines 7 and 8, a GIN is run over both graphs to get separate node embeddings for the image and text graphs.
Lines 10 and 11 use the `to_dense_batch` function in PyG. This is used because each batch contains graphs of different sizes (number of nodes). In order to process all graphs together, we need to make the number of nodes equal for all graphs. This is done by using `to_dense_batch` to construct “fake nodes” which are not connected to any other node in order to equalize the number of nodes in each graph. The function returns a mask, which indicates which nodes are real and which are fake.
Line 18 constructs  like the equation shown above. The mask is used so that while normalizing , we don’t assign some probability to fake nodes as well.
Finally, line 23 projects the text embeddings into image space, and line 26 constructs the final multimodal representation of the graph by averaging over the concatenation of the two embeddings.

## Results

We’ll now run a fully implemented MGN model over a subset of the CLEVR dataset, a diagnostic dataset of 3D shapes which tests visual and linguistic reasoning. As seen below, our model is able to answer complex questions based on the images.

The question for this image was “There is a tiny matte thing that is left of the big gray cylinder and in front of the cyan metal cylinder; what color is it?” Our model predicts “Purple,” which is correct.

The question for this image was “Is there anything else that has the same color as the large shiny cube?” Our model predicts “No,” which is incorrect.

The question for this image was “There is a small blue block; are there any spheres to the left of it?” Our model predicts “Yes,” which is correct.

To play with our implementation, you can go to the following Colab. This Colab implements the MGN model, trains it over a small subset of the CLEVR dataset, and performs inference on a small dataset. You can play around with the inputs to the model and see how the model performs.
For completeness, we also show the quantitative results of the model from the original paper on the entire CLEVR dataset:

The paper synthetically generates two image datasets — A and B — with each image containing just one object. In dataset A, all cubes are gray, blue, brown, or yellow and all cylinders are red, green, purple, or cyan. In B, cubes and cylinders swap color palettes.

## Conclusion

We’ve seen how graphs can be a powerful abstraction for representing both text and images as well as the relationships between them. Using ideas from graph machine learning, such as GIN, we generated node embeddings from our input. Finally, we combined these embeddings to create a multimodal representation that our downstream model could use to answer questions about an image. Overall, we hope to have illustrated the interesting challenges of multimodal tasks such as VQA and the versatility of graph learning methods in solving them.

### References

[1] Saqur, Raeid, and Karthik Narasimhan. “Multimodal graph networks for compositional generalization in visual question answering.” Advances in Neural Information Processing Systems (2020).
[2] Marino, Kenneth, et al. “Ok-vqa: A visual question answering benchmark requiring external knowledge.” Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019.
[3] He, Kaiming, et al. “Mask r-cnn.” Proceedings of the IEEE international conference on computer vision. 2017.
[4] https://perfectial.com/blog/graph-neural-networks-and-graph-convolutional-networks/
[5] Xu, Keyulu, et al. “How powerful are graph neural networks?.” arXiv preprint arXiv:1810.00826 (2018).