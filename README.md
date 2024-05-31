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

# Experiment 2: Using Two Agents - Sequential Dialogue
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
```

# Experiment 3 - Multi Agents & Cross-referencing Responses
Now, We have four agents who are brainstorming for a new tv show. There's Agent Director, Agent Character, Agent Screenplay, Agent Producer.

```bash
import ollama

def agent_story(prompt):
    """Generates a story idea based on the input prompt."""
    try:
        response = ollama.generate(model='llama3', prompt=f"Generate a story idea about: {prompt}")
        return response['response']
    except Exception as e:
        return f"An error occurred in agent_story: {e}"

def agent_character(story_idea):
    """Develops characters based on the story idea."""
    try:
        response = ollama.generate(model='llama3', prompt=f"Create detailed character backgrounds for the story: {story_idea}")
        return response['response']
    except Exception as e:
        return f"An error occurred in agent_character: {e}"

def agent_dialogue(story_idea):
    """Writes dialogues or short paragraphs based on the storyline."""
    try:
        response = ollama.generate(model='llama3', prompt=f"Write a short paragraph or dialogue for the story: {story_idea}")
        return response['response']
    except Exception as e:
        return f"An error occurred in agent_dialogue: {e}"

def agent_producer(story, characters, dialogue):
    """Compiles responses from all agents and provides a critique or appreciation."""
    unified_response = f"Story Idea: {story}\n\nCharacters: {characters}\n\nDialogue: {dialogue}"
    try:
        critique_prompt = f"Provide a critique or appreciation for the following story idea and its elements:\n\n{unified_response}"
        response = ollama.generate(model='llama3', prompt=critique_prompt)
        return response['response']
    except Exception as e:
        return f"An error occurred in agent_producer: {e}"

if __name__ == "__main__":
    user_prompt = input("Enter a theme or idea for the story: ")
    
    # Generate story idea
    story_idea = agent_story(user_prompt)
    print(f"Agent Story Response:\n{story_idea}\n")
    
    # Generate character backgrounds
    character_backgrounds = agent_character(story_idea)
    print(f"Agent Character Response:\n{character_backgrounds}\n")
    
    # Generate dialogues or short paragraphs
    dialogue = agent_dialogue(story_idea)
    print(f"Agent Dialogue Response:\n{dialogue}\n")
    
    # Producer's critique or appreciation
    producer_response = agent_producer(story_idea, character_backgrounds, dialogue)
    print(f"Agent Producer Response:\n{producer_response}\n")
```

### Example Output: Prompt
Enter Prompt: Toil of Modern Men balancing Traditional Men and Modern Women in 21st Century

Agent Director Response:
```plaintext
What an intriguing concept! Here's a potential story idea:

**Title:** "The Weight of Progress"

**Premise:** Meet Jack, a well-intentioned but self-proclaimed "feminist man" who finds himself struggling to balance
his passion for women's rights with the weight of societal expectations, family pressures, and his own personal doubts.
As he navigates this internal conflict, Jack must confront the complexities of being a male ally in a movement that often questions his motives.

**Storyline:** Jack's journey begins when his girlfriend, Sarah, a prominent feminist activist, asks him to join her at a
conference on gender equality. Initially excited to support her cause, Jack soon discovers that many attendees view him with
skepticism, perceiving him as an outsider trying to co-opt the movement for his own ego boost.

As he grapples with these criticisms, Jack's relationships with Sarah and their friends begin to fray. His family, particularly
his traditionalist father, disapprove of his involvement in feminism, seeing it as a threat to "traditional" values. Meanwhile,
Jack's own insecurities surface: Is he truly committed to the cause, or is he just trying to impress Sarah?

As tensions escalate, Jack finds himself torn between his desire to be a supportive partner and his need to prove himself as a
genuine ally. He begins to question whether his involvement in feminism is genuinely empowering women or simply perpetuating paternalistic attitudes.

**Themes:**

1. The burden of being a male ally: How can men support the feminist movement without co-opting its message or
undermining women's agency?
2. The complexities of intersectionality: Jack's struggles highlight the need for nuanced understanding of how different forms of
oppression intersect and impact individual experiences.
3. Self-reflection and personal growth: Through his journey, Jack must confront his own biases, insecurities, and privileges to
become a more effective ally.

**Genre:** Contemporary drama/romance

**Tone:** Thought-provoking, introspective, and emotionally charged, with a touch of humor and wit to balance the weighty themes.

This story idea explores the challenges faced by male allies in the feminist movement, encouraging readers to reflect on their
own roles in promoting gender equality. How do you think Jack's character development would unfold?
```

Agent Character Response:
```plaintext
What a fascinating premise! Let me help you develop detailed character backgrounds for Jack:

**Name:** Jackson "Jack" Thompson

**Age:** 28

**Occupation:** Marketing specialist, part-time freelance writer

**Personality:**

1. **Well-intentioned**: Jack genuinely wants to support the feminist movement and women's rights.
2. **Eager to learn**: He's curious about social justice issues and willing to listen and educate himself.
3. **Insecure**: Jack struggles with self-doubt, seeking validation from others (especially Sarah) and uncertain of his role in the movement.
4. **Passionate**: He's enthusiastic about causes he believes in and enjoys debating topics that spark his interest.
5. **Self-deprecating**: Jack often underestimates himself, thinking that others are more qualified or knowledgeable than him.

**Background:**

1. **Upbringing:** Jack grew up in a traditionalist family where gender roles were strictly defined. His father, a conservative
businessman, valued "masculinity" and expected men to be breadwinners.
2. **Education:** Jack earned a degree in marketing from a local university. Although he excelled academically, his passion for
social justice issues wasn't fully explored until later in life.
3. **Relationships:**
        * **Sarah**: His girlfriend of three years, Sarah is a prominent feminist activist and the driving force behind his
involvement in the movement. Jack admires her conviction and strength.
        * **Family:** Jack's family is skeptical of his involvement in feminism. His father, in particular, disapproves,
viewing it as a threat to traditional values.
4. **Values:**
        * **Empathy**: Jack believes in understanding others' perspectives, even if he doesn't agree with them.
        * **Equality**: He advocates for equal rights and opportunities for all individuals, regardless of gender.

**Goals:**

1. **Prove himself**: Jack wants to demonstrate his commitment to the feminist movement and prove that he's not just a "token" male ally.
2. **Support Sarah**: He hopes to be a reliable partner for Sarah, helping her in her activism and showing appreciation for her passions.
3. **Self-improvement**: Jack aims to become a more effective ally by educating himself on intersectionality and the complexities of gender equality.

**Conflict:**

1. **Internal struggles**: Jack's insecurities about his role in the movement and his desire to prove himself lead to internal conflicts.
2. **External pressures**: He faces skepticism from some feminist activists, who view him as an outsider trying to co-opt their message.
His family's disapproval also adds to the pressure.

By understanding Jack's character background, we can better appreciate the challenges he faces in "The Weight of Progress." As
he navigates these complexities, he'll have opportunities for growth, self-reflection, and personal development.
```

Agent Screenplay Response:
```plaintext
Here's a potential short paragraph or dialogue for the story:

**Scene:** Jack and Sarah are having dinner at her family's home, where they're hosting a gathering of feminist activists.
As they mingle with friends and acquaintances, Jack notices the skeptical glances and whispered conversations.

"Hey, what's going on?" Jack asks Sarah, feeling uneasy.

Sarah looks around the room before answering quietly, "They think you're just trying to impress me or gain brownie points for being a 'good guy.'"

Jack's eyes widen in surprise. "That's not fair! I'm here because I genuinely care about this cause."

Sarah places a reassuring hand on his arm. "I know that, Jack. But sometimes, men's involvement can be seen as tokenistic.
It's like, what are you doing to actively dismantle the patriarchy, instead of just showing up and saying 'Hey, look at me, I'm a feminist!'"

Jack takes a deep breath, processing her words. "You're right. I need to prove myself, not just my intentions."

**Themes:** This scene highlights the tension between Jack's genuine desire to support Sarah and his recognition that his involvement
is not without its challenges. It also touches on the theme of intersectionality, as Sarah cautions Jack about the importance of taking
concrete actions rather than just paying lip service.

This dialogue sets the stage for Jack's character development, as he grapples with his own doubts, insecurities, and privileges. The
conversation also foreshadows the conflicts he'll face as he navigates the complexities of being a male ally in the feminist movement.
```

Agent Producer Response:
```plaintext
What an intriguing story idea! I appreciate the thought-provoking themes and the character development potential. Here's my critique:

**Strengths:**

1. **Timely and relevant**: The concept of "The Weight of Progress" is timely, given the current social climate and the importance of
intersectionality in social justice movements.
2. **Complex character**: Jack's character has depth, with both positive traits (well-intentioned, eager to learn) and flaws
(insecure, self-doubting). This complexity makes him relatable and allows for growth throughout the story.
3. **Thought-provoking themes**: The exploration of intersectionality, paternalistic attitudes, and the burden of being a male ally in
the feminist movement will likely spark important discussions among readers.

**Weaknesses:**

1. **Overemphasis on Jack's internal struggles**: While it's essential to explore Jack's character development, the story might benefit
from a more balanced approach between his inner conflicts and the external challenges he faces.
2. **Potential for didacticism**: The themes and messages in this story could become heavy-handed or didactic if not handled carefully.
It's crucial to strike a balance between conveying important ideas and allowing readers to draw their own conclusions.

**Suggestions:**

1. **Introduce conflicts earlier**: To create more tension, consider introducing external conflicts (e.g., disagreements with Sarah or
her friends, family disapproval) earlier in the story.
2. **Vary pacing and tone**: Balance the introspective moments with more action-oriented scenes to maintain a engaging pace.
3. **Explore the relationships between characters**: Delve deeper into Jack's relationships with Sarah, his family, and other
characters to create a richer narrative.

**Overall:**

"The Weight of Progress" has immense potential for character growth, thematic exploration, and social commentary. By addressing some
of the suggested weaknesses and areas for improvement, you can craft a compelling story that resonates with readers.
