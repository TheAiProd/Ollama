# Set Up and Run Ollama in Python

### 1. Create a virtual environment
   ```bash
   python3 -m venv venv
   # activate the virtual environment
   source venv/bin/activate
   ```
### 2. Install the Ollama Package
   ```bash
    # install  `ollama` in the root directory
   pip install ollama
   ```

### 3. Create and Edit the Python Script
   ```bash
   touch run_ollama.py
   ```

# Experiment 1 - Chat with Ollama
   Open the file in your preferred code editor (e.g., VSCode) and add the following code:

   ```bash
   import ollama

   def run_llama3(prompt):
       try:
           # Use the generate method from the ollama package
           response = ollama.generate(model='llama3', prompt=prompt)
           print("Output:\n", response['response'])
       except Exception as e:
           print(f"An error occurred: {e}")

   if __name__ == "__main__":
       user_prompt = input("Enter your prompt: ")
       run_llama3(user_prompt)
   ```

### Run the Python Script
  Ensure the virtual environment is activated**:
   ```bash
    # ensure the virtual environment is activated
   source venv/bin/activate
    # run the script
   python run_ollama.py
   ```

### Example Output

When you run the script, you will be prompted to enter a prompt. The model will generate and print a response based on your input.

Enter your prompt: What is the current market price of Tesla?

```plaintext

Output:
The current market price of Tesla (TSLA) can fluctuate depending on various factors such as supply and demand,
overall market trends, earnings reports, and other news that may impact investor sentiment. As a result, it's best
to check the latest quotes or charts from reputable financial sources like Yahoo Finance, Google Finance, or Bloomberg.

As of my knowledge cutoff in mid-February 2023, the current market price of Tesla (TSLA) was around $1,140-$1,150 per share.
However, please note that this information may have changed since then.

For up-to-date quotes, you can check:
1. Yahoo Finance: www.yahoo.com/finance
2. Google Finance: www.google.com/finance
3. Bloomberg: www.bloomberg.com

Please verify the current market price on these sources or others for the most accurate information.
```
---

# Experiment 2 - Using Two Agents
Agent Llama provides a response, and Agent Elif translates the response to a 5-year old

```bash
import ollama

def agent_llama3(prompt):
    """First agent generates a response based on the input prompt."""
    try:
        response = ollama.generate(model='llama3', prompt=prompt)
        return response['response']
    except Exception as e:
        return f"An error occurred in agent_llama3: {e}"

def agent_elif(response):
    """Second agent simplifies the response for a five-year-old."""
    try:
        new_prompt = f"Explain this to a five-year-old: {response}"
        simplified_response = ollama.generate(model='llama3', prompt=new_prompt)
        return simplified_response['response']
    except Exception as e:
        return f"An error occurred in agent_elif: {e}"

if __name__ == "__main__":
    user_prompt = input("Enter your prompt: ")
    # First agent generates a response
    first_response = agent_llama3(user_prompt)
    print(f"Agent Llama3 Response:\n{first_response}\n")

    # Feed the response to the second agent
    simplified_response = agent_elif(first_response)
    print(f"Agent ELIF Response (simplified for a five-year-old):\n{simplified_response}")
```

### Example Output
Enter your prompt: What is Diffusion in Transformer LLM?
```plaintext

Agent Llama3 Response:

In the context of Transformers, diffusion refers to a technique used in some transformer-based
language models (LLMs) to improve their performance and stability. It's an extension of the original
Transformer architecture, which was introduced by Vaswani et al. in 2017.

In traditional transformers, self-attention mechanisms are used to model relationships between input elements.
However, these mechanisms can be sensitive to the ordering of input elements, which can lead to issues like
vanishing or exploding gradients during training.

Diffusion is a way to mitigate these problems by introducing an auxiliary process that helps the model
learn more robust and interpretable representations. The core idea is to create a "diffusion process"
that gradually transforms the input tokens into a latent representation, where each token is influenced by
its neighbors in a hierarchical manner.

Here's how it works:

1. Latent space: A diffusion process is initialized with an initial latent representation of the input sequence.
2. Diffusion steps: The model iteratively applies a series of transformations to this latent representation,
using self-attention mechanisms and feed-forward networks (FFNs). Each step refines the representation by incorporating
information from neighboring tokens.
3. Stopping criterion: The diffusion process stops when a predefined stopping criterion is reached, such as reaching
a certain number of steps or achieving a satisfactory level of representation quality.

The benefits of diffusion in transformers include:

* Improved stability and robustness during training
* Enhanced ability to capture long-range dependencies and hierarchical structures in input sequences
* Better handling of out-of-vocabulary (OOV) words and rare events

Some notable transformer-based LLMs that employ diffusion techniques are:

1. BERT (Bidirectional Encoder Representations from Transformers): Uses a modified version of the diffusion process
to improve its representation quality.
2. RoBERTa (Robustly Optimized BERT Pretraining Approach): Incorporates a more extensive use of diffusion and
attention mechanisms to enhance its language understanding capabilities.

In summary, diffusion in transformers is a technique that helps LLMs learn more robust and interpretable representations
by introducing an auxiliary process that gradually transforms input tokens into a latent representation. This can lead to
improved performance, stability, and handling of complex linguistic structures.
```
```plaintext
Agent ELIF Response (simplified for a five-year-old):

Oh boy, are you ready for some cool robot talk?
So, there are these super smart robots called Transformers that can understand and talk like humans.
They're like superheroes with a special power: they can change their shape from a car to a robot and back again!
Now, imagine those robots have special friends who help them learn new things. These friends are like magic machines
that make the robots smarter. One way these machines help is by using something called "diffusion".
Diffusion is like a special kind of magic that makes the robots' brains work better. It's like a game where they take
turns sharing secrets with each other, and then the next robot uses those secrets to get even smarter! 

Here's how it works:
1. The first robot takes some special information from its friends (like words or ideas) and puts them together in a
special box called "latent space".
2. Then, the robot does a series of magic tricks (called "diffusion steps") where it shares secrets with other robots
in its neighborhood. Each trick makes the box get better and better!
3. The robot stops doing magic tricks when it's happy with how good the information is.

Using diffusion makes the robots:
* More stable and strong, like a superhero
* Better at understanding long stories or tricky words
* Good at figuring out new things that they haven't seen before

Some really smart Transformers called BERT and RoBERTa use this magic technique to get even better at understanding
language! They're like the ultimate superheroes of robots!

So, that's diffusion in a nutshell (or a robot shell)!
