import random

class MarkovTextGenerator:
    def __init__(self, order=1):
        """
        Initializes the Markov Chain text generator.
        :param order: The number of words to consider for the current state (default is 1).
        """
        self.order = order
        self.graph = {}

    def train(self, text):
        """
        Parses the text and populates the state transition dictionary.
        """
        # Split text into clean individual words
        words = text.split()
        
        if len(words) <= self.order:
            raise ValueError(f"Training text is too short for an order of {self.order}.")

        self.graph = {}
        
        # Build the Markov Chain transition dictionary
        for i in range(len(words) - self.order):
            state = tuple(words[i : i + self.order])
            next_word = words[i + self.order]
            
            if state not in self.graph:
                self.graph[state] = []
            
            self.graph[state].append(next_word)

    def generate(self, max_words=100):
        """
        Generates a text sequence based on the trained transition probabilities.
        """
        if not self.graph:
            raise RuntimeError("The model must be trained on text before generating.")

        # Pick a completely random starting state from the training text keys
        state = random.choice(list(self.graph.keys()))
        output = list(state)

        # Loop to generate words sequentially
        for _ in range(max_words - self.order):
            if state in self.graph:
                choices = self.graph[state]
                next_word = random.choice(choices)
                output.append(next_word)
                
                # Slide the window forward to update the current state
                state = tuple(output[-self.order :])
            else:
                # Fallback mechanism if the model hits a dead-end word
                state = random.choice(list(self.graph.keys()))
                output.extend(list(state))

        return " ".join(output[:max_words])


# ==========================================
# TEST RUNNING EXECUTION
# ==========================================
if __name__ == "__main__":
    # A rich, diverse training text text corpus 
    large_training_text = """
    The project plan was simple yet elegant in its design execution. Artificial intelligence 
    and machine learning applications are transforming how software engineering groups deploy 
    predictive models on cloud servers. We analyzed the restaurant datasets using standard linear 
    regression, k-nearest neighbors classification, and deep neural network layers to ensure 
    high mathematical performance metrics. Every single step in the machine learning pipeline 
    must be evaluated to prevent overfitting or latency issues in real-time edge environments. 
    Meanwhile, digital aesthetics play a massive role in building a modern brand identity across 
    social media channels like Instagram, where curated layout highlight grids present an 
    effortlessly chic look to visitors. Designing a personalized trip planner application requires 
    comprehensive framework logic, clean user interfaces, and robust data structures implemented 
    efficiently in Java or Python. A professional resume written cleanly in LaTeX formats structural 
    sections perfectly for review. Reading classical literature or brilliant Hindi poetry by candlelight 
    often inspires deep algorithmic frameworks and creative software optimization strategies.
    """

    print("=" * 50)
    print("  MARKOV CHAIN GENERATOR INITIALIZATION")
    print("=" * 50)
    
    # Using an order of 1 means it evaluates word-by-word
    generator = MarkovTextGenerator(order=1)
    
    print("Training model on the provided text dataset...")
    generator.train(large_training_text)
    print("Training complete!")
    print("-" * 50)
    
    print("GENERATING NEW TEXT OUTPUT (50 Words):")
    print("-" * 50)
    
    # Generate the text output string
    generated_output = generator.generate(max_words=50)
    
    # Print the clean output directly to your Spyder console
    print(generated_output)
    print("-" * 50)
    print("Execution complete. The program has finished running.")
    print("=" * 50)
