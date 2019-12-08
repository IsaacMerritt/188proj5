import nn
import numpy as np

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.w, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        if nn.as_scalar(self.run(x)) >= 0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***" 
        while True:
            mistake = 0
            for x, y in dataset.iterate_once(1):
                class_prediction = self.get_prediction(x)
                if class_prediction != nn.as_scalar(y):
                    mistake = 1
                    self.w.update(x, nn.as_scalar(y))
            if not mistake:
                break

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        self.learning_rate = .1
        self.batch_size_ratio = .1 # batch size is ratio * len(dataset)
        self.threshold = 0.002
        self.num_layers = 2
        self.layer_sizes = [300, 300]
        self.hidden_layers = []

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        y = x
        for layer in self.hidden_layers:
            y = nn.Linear(y, layer[0])
            y = nn.AddBias(y, layer[1])
            y = nn.ReLU(y)
            y = nn.Linear(y, layer[2])
            y = nn.AddBias(y, layer[3])
        return y


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predictions = self.run(x)
        return nn.SquareLoss(predictions, y)


    def get_parameters(self):
            params = []
            for layer in self.hidden_layers:
                for param in layer:
                    params.append(param)
            return params

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        batch_size = int(self.batch_size_ratio * dataset.x.shape[0])
        while len(dataset.x) % batch_size != 0:
            batch_size += 1
        self.hidden_layers = [
            ([
                nn.Parameter(dataset.x.shape[1], self.layer_sizes[i]),
                nn.Parameter(1,self.layer_sizes[i]),
                nn.Parameter(self.layer_sizes[i], dataset.x.shape[1]),
                nn.Parameter(1,1)
            ])
            for i in range(self.num_layers)
        ]
        while True:
            losses = []
            for x, y in dataset.iterate_once(batch_size):
                loss = self.get_loss(x,y)
                params = self.get_parameters()
                gradients = nn.gradients(loss, params)
                for i in range(len(params)):
                    param = params[i]
                    param.update(gradients[i], -self.learning_rate)
                losses.append(nn.as_scalar(loss))
            if np.mean(losses) < self.threshold:
                break



class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.learning_rate = .25
        self.batch_size_ratio = .1
        self.threshold = 0.973
        self.hidden_layers = [
            [
                nn.Parameter(784, 250),
                nn.Parameter(1, 250),
                nn.Parameter(250, 784),
                nn.Parameter(1,784)
            ],
            [
                nn.Parameter(784, 50),
                nn.Parameter(1, 50),
                nn.Parameter(50, 10),
                nn.Parameter(1,10)
            ],
        ]

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        y = x
        for layer in self.hidden_layers:
            y = nn.Linear(y, layer[0])
            y = nn.AddBias(y, layer[1])
            y = nn.ReLU(y)
            y = nn.Linear(y, layer[2])
            y = nn.AddBias(y, layer[3])
        return y
        

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predictions = self.run(x)
        return nn.SoftmaxLoss(predictions, y)


    def get_parameters(self):
        params = []
        for layer in self.hidden_layers:
            for param in layer:
                params.append(param)
        return params

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        batch_size = int(self.batch_size_ratio * dataset.x.shape[0])
        while len(dataset.x) % batch_size != 0:
            batch_size += 1
        while True:
            for x, y in dataset.iterate_once(batch_size):
                
                loss = self.get_loss(x,y)
                params = self.get_parameters()
                gradients = nn.gradients(loss, params)
                for i in range(len(params)):
                    param = params[i]
                    param.update(gradients[i], -self.learning_rate)
            if dataset.get_validation_accuracy() > self.threshold:
                break

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.learning_rate = .001
        self.batch_size = 1
        self.threshold = 0.82
        self.hidden_size = 800

        self.hidden_bias = nn.Parameter(self.batch_size,self.num_chars)
        self.hidden_weights = nn.Parameter(self.num_chars, self.hidden_size)
        self.b0 = nn.Parameter(self.batch_size,self.hidden_size)
        self.weights = nn.Parameter(self.hidden_size, self.num_chars)
        self.b1 = nn.Parameter(self.batch_size,self.num_chars)
        self.final_weights = nn.Parameter(self.num_chars, len(self.languages))




    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"

        hidden_state = xs[0]
        hidden_layer = nn.AddBias(hidden_state, self.hidden_bias)

        for x in xs[1:]:
            layer = x
            #layer_weighted = nn.Linear(layer, self.hidden_weights)
            #hidden_layer_weighted = nn.Linear(hidden_layer,self.weights)
            layer = nn.Add(layer, hidden_layer)
            layer = nn.Linear(layer, self.hidden_weights)
            layer = nn.AddBias(layer, self.b0)
            layer = nn.ReLU(layer)
            layer = nn.Linear(layer, self.weights)
            layer = nn.AddBias(layer, self.b1)
            hidden_layer = layer
        
        layer = nn.Linear(hidden_layer, self.hidden_weights)
        layer = nn.AddBias(layer, self.b0)
        layer = nn.ReLU(layer)
        layer = nn.Linear(layer, self.weights)
        layer = nn.AddBias(layer, self.b1)
        layer = nn.ReLU(layer)

        y_predictions = nn.Linear(layer, self.final_weights)
        return y_predictions


    def get_parameters(self):
        return [self.weights,self.b0,self.hidden_weights,self.b1,self.final_weights]



    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predictions = self.run(xs)
        return nn.SoftmaxLoss(predictions, y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                
                loss = self.get_loss(x,y)
                params = self.get_parameters()
                gradients = nn.gradients(loss, params)
                for i in range(len(params)):
                    param = params[i]
                    param.update(gradients[i], -self.learning_rate)
            if dataset.get_validation_accuracy() > self.threshold:
                break
