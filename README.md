# Generative-AI-Beginner-to-Pro-with-OpenAI-Azure-OpenAI-Notes

**I) AI Concepts and Workloads - (For Absolute Beginners- Optional)**

**A) What is AI ?**

**B) History of AI**

**C) Benefits of Artificial Intelligence (AI)**

**D) Types of AI Workloads**

**E) AI vs ML vs DL**

**II) Machine Learning Foundations (For Absolute Beginners-Optional)**

**A) Machine Learning - Real World Example**

**B) Machine Learning - Key Terminologies**

**C) What is Machine Learning ?**

**D) Types of Machine Learning**

**E) What is Supervised Machine Learning ?**

**F) What is Classification in SML ?**

**G) What is Regression in SML ?**

**H) What is Unsupervised Machine Learning ?**

**I) What is Re-inforcement Machine Learning**

**J) What is a Jupyter Notebook**

**K) Demo: Install Anaconda**

**L) Demo: Understanding the IRIS Dataset**

**M) Demo: Creating & Training your ML Model**

**III) Deep Learning Foundations (For Absolute Beginners- Optional)**

**A) What is Deep Learning ?**

**B) What is a Neural Network ?**

**C) Deep Learning Models**

**D) What is a Transformer Model?**

**E) Demo: GANs - Deep Fake Video**

**F) Demo- Creating & Training Deep Learning Model**

**IV) Generative AI Foundations (For Beginners)**

**A) What is Generative AI ?**

**B) Predictive AI vs Generative AI**

**C) What is LLM ?**

**D) What is Embedding ?**

**E) What is a Vector Database ?**

**F) Embedding vs Vector Database**

**G) What is Retriever Augmented Generation (RAG) ?**

**H) What is Langchian ?**

**I) Role of Langchain in RAG**

**J) Prompt Engineering & Fine Tuning**

**V) AI Infrastructure**

**A) What is a GPU ?**

**B) Demo: CPU Vs GPU**

**C) What is RDMA Cluster Network**

**VI) OpenAI / ChatGPT / API**

**A) What is OpenAI ?**

**B) What is ChatGPT ?**

**C) Demo on chatgpt**

**D) Time to reach 100M users**

**E) Get an understanding of Various OpenAI Models**

**F) GPT-3 vs GPT-4**

**G) New: What is GPT-4o ?**

**H) New: Demo: GPT-4 Vs GPT-4o**

**I) New: What is GPT-4o Mini?**

**J) What are tokens ?**

**K) Pricing Model for ChatGPT**

**L) Demo: How to make API Calls with OPENAI APIs**

**M) Demo: Make a simple API Call**

**N) Demo: How to create Embeddings ?**

**O) Demo: Image generation using DALL·E in the API**

**P) Demo: Convert Speech to Text**

**Q) New: What is OpenAI O1 Model ?**

**R) Demo: Compare GPT-4o vs OpenAI O1**

**VII) Azure OpenAI Foundations**

**A) Azure OpenAI - Intro**

**B) What is Azure OpenAI**

**C) History behind Azure OpenAI**

**D) Models available with Azure OpenAI(Regions)**

**E) Limits & Quotas - Important Consideration**

**F) How Pricing Works in Azure OpenAI**

**G) Demo: Setup Azure OpenAI Service**

**H) What is Azure Open AI Studio ?**

**I) Demo: Azure OpenAI Studio Walkthrough**

**J) Chat Playground - Demo: Create a Deployment of Chats Playground**

**K) Understand the Chat Playground**

**L) Demo: Deploy a Webapp from the Playground**

**M) DALL-E PlayGround - Demo on Generating Images**

**N) What is Completions Playground ?**

**O) Demo: Completions Playground**

**VIII) AI Foundry - Covered in his AI Foundry course; Please refer to that Notes**

**IX) Azure OpenAI - Making API Calls**

**A) API Calls - Intro**

**B) OpenAI API Calls Vs Azure OpenAI API Calls**

**C) Demo: Create a New Azure OpenAI Service**

**D) Demo: Get the Values of Endpoint URL & API Keys**

**E) Demo: Create an azureopenai.env File**

**F) Demo: Get the value of api_version**

**G) Demo: Create a New Deployment of Chats Completion**

**H) Demo: Make a Simple Azure OpenAI API Call**

**No need Notes for rest of the Course; They are all already what has been done for AWMS Working and AI Foundry Course, similar to create Index and retrieving data; Course is always with you; Refer it**



# **I) AI Concepts and Workloads - (For Absolute Beginners- Optional)**

# **A) What is AI ?**

Hi folks, welcome back. So we need to start with the one-million-dollar question: What is AI? Right now we are looking to go deep into the certifications for artificial intelligence, but what if we actually don’t understand what AI really is? How should we start learning these concepts? That’s why it is very important to first understand what AI actually means. It’s always good to break down the words. So let’s break the term Artificial Intelligence and understand the meaning.

First, what do we mean by artificial? Artificial is something that is not natural. When we talk about something natural, we mean something that exists in nature on its own. But when we talk about something artificial, it is man-made. So the first thing to remember is that artificial refers to something created by humans.

Next, what is intelligence? It is a very simple word, but it carries a deep meaning. Intelligence is the ability to acquire and apply knowledge and skills. So when we combine the two terms, we get Artificial Intelligence. This means intelligence that is man-made, not natural. Humans have natural intelligence, but machines and computers have artificial intelligence.

If we look at the complete definition of AI, it is the ability or capability of a computer system to mimic human-like cognitive functions. Now you might ask, what are cognitive functions? Cognitive mainly involves three things: knowing, learning, and understanding. So when we try to make computers smart enough to have human-like cognitive abilities—where computers can understand, learn, and know things—we are essentially talking about AI. In short, AI is the ability of a computer system to mimic human-like cognitive functions.

In the next part, let’s quickly look at a demo of AI. I thought it would be good to give you a quick example of what AI looks like, especially if you are new to this field. So let’s take a look.

Introducing Phantom—the most advanced chessboard in the world. It brings you the infinite possibilities of online chess with the engaging experience of a physical set. Phantom is also the smartest board ever. It allows you to play against any human on Earth remotely, moving the pieces using only your voice. For example, you can simply say, “Knight G4,” and the board responds. You can also play against its human-like AI, which continuously adapts to your playing level. The pieces can even replay the most famous games in history. Phantom brings back all the little details that make chess great.

This is a perfect example of artificial intelligence. If you noticed, the person controlling the board was simply giving voice commands. In the world of AI, this falls under NLP (Natural Language Processing). You just say “move the knight to G4,” and based on your speech, the piece moves. The system also includes a human-like AI that adapts to your playing level. The chessboard is so intelligent that it can understand whether you are an expert or a beginner simply based on the type of moves you play. Within the first few moves, it can gauge your level and adjust its own strategy accordingly. A perfect demonstration of AI in action.

# **B) History of AI**

So my dad always says that whenever you have to start with a new subject, you should always begin with its history. That’s exactly what I’ve done here. I’ve created a complete timeline of artificial intelligence for you, and you’ll actually be surprised to see that artificial intelligence has existed for ages. We’re talking all the way back to the 1950s. In 1950, a British mathematician named Alan Turing published a groundbreaking paper titled Computing Machinery and Intelligence. This is where the idea of making computers as intelligent as humans truly began. He also introduced the famous Turing Test, a simple method to determine whether a machine can demonstrate human intelligence.

Then we move to 1956, when John McCarthy coined the term Artificial Intelligence at the first-ever AI conference held at Dartmouth College. This is why he is often called the father of artificial intelligence. From there, things began to mature further in the 1960s. In this decade, Frank Rosenblatt built the Mark I Perceptron, the first computer based on a neural network. Don’t worry if you don’t know much about neural networks yet―we will deep dive into them later. For now, think of neural networks as an attempt by scientists to mimic how the human brain works using neurons. These early systems mostly worked on trial and error: if something went wrong, the algorithm was improved and refined.

Then we jump to the 1980s, when neural networks became more mature. This is when the concept of backpropagation emerged. Backpropagation is a gradient estimation method used to train neural networks by adjusting their internal weights. It was a huge step forward in making machine learning more effective and practical.

In 1997, a major milestone shocked the world. IBM’s Deep Blue defeated Garry Kasparov, the reigning world chess champion. It was global news because it was the first time a computer had beaten a world champion in chess, proving how far AI had progressed.

Advancing to 2011, AI had another major moment with the game show Jeopardy! Instead of being asked questions, contestants are given clues in the form of answers and must respond with the correct question. IBM’s Watson competed against champions Ken Jennings and Brad Rutter—and Watson won. It demonstrated the enormous potential of AI in understanding natural language and processing vast amounts of information.

By 2015, things were heating up even more. The Chinese tech giant Baidu introduced Minerva, a supercomputer that used advanced deep neural networks. Its image identification and categorization abilities exceeded the accuracy of the average human. This was one of the early signals that AI could outperform humans in certain cognitive tasks.

In 2016, another historic moment arrived with the ancient board game Go. Go is a complex strategy game where the objective is to capture more territory than the opponent. DeepMind’s AlphaGo, powered by deep neural networks, defeated world champion Lee Sedol in a five-game match. Within those five games, AlphaGo proved that AI could master even the most complex strategy-based human games.

Then we arrive at the 2020s, which I always call the true game changer. AI was always there, but what completely changed its face was Generative AI. OpenAI released GPT-3, one of the world’s most sophisticated language processing models capable of generating human-like text. This is fundamentally different from predictive AI. Generative AI can understand context and create original content—stories, poems, essays, letters—just from a simple prompt. That’s why I say this era is transformative. From 2020 onwards, you will see rapid evolution in this field, and this is where companies are now putting their R&D budgets.

So with this, I think you now have a solid understanding of the entire AI timeline.

# **C) Benefits of Artificial Intelligence (AI)**

Hello and welcome. After understanding what AI is and exploring the AI timeline, it's now time to look into the benefits of artificial intelligence. Why is AI so important, and why are we learning about it today? The key thing to remember is that AI has been around us for many years, as we saw in the timeline, and its impact continues to grow.

One of the most important benefits is no human error. Humans naturally make mistakes, and many major outages or system failures in the real world have happened due to human error. AI systems, on the other hand, are smart and intelligent machines that follow precise instructions and algorithms. Since the tasks are being performed by computers and not humans, the likelihood of errors significantly reduces.

Another major advantage is 24×7 availability. Humans need to eat, sleep, rest, and take breaks in order to function properly, but machines do not. AI systems can work 24 hours a day, 7 days a week, and 365 days a year without getting tired. They are always available and do not require downtime like humans do.

Next is the fact that humans can be biased, whereas machines are not. Humans may be emotionally inclined or unfair due to personal beliefs or preferences. Machines, however, make decisions purely based on the data, algorithms, and models they are trained on, not based on emotions. This leads to more unbiased and consistent decision-making.

AI also enables quicker decision-making. While the human brain is powerful, processing large volumes of data manually can take a lot of time. AI systems excel at parallel processing. You can feed huge amounts of data into a machine, and it can quickly analyze, comprehend, and make decisions much faster than a human could.

AI also helps in reducing risks—maybe not “no risks,” but definitely lesser risks. Since machines do not make human errors and do not carry emotional biases, organizations can significantly lower operational and decision-making risks by relying on AI.

One area where AI has shown tremendous progress is healthcare. AI systems are being used to diagnose diseases, predict health outcomes, and even recommend treatment plans. This has elevated the quality of medical assistance and opened new possibilities in patient care, diagnosis, and precision medicine.

Another key benefit is the ability to manage recurring tasks effectively. In many companies, employees are assigned repetitive daily tasks—like running the same script every morning at 9 AM. Over time, a human will naturally feel bored, frustrated, or unmotivated because the task is mundane. Machines, however, do not complain, do not get bored, and do not experience fatigue. They can perform repetitive tasks consistently and accurately without any emotional response, making them ideal for automation.

While AI offers many more benefits beyond these, the points covered here represent the most essential advantages you should keep in mind as you continue learning about artificial intelligence.

# **D) Types of AI Workloads**

Now that we’ve already covered the fundamentals of AI, it’s time to look at the various workloads that exist within artificial intelligence. When we talk about AI workloads, we are basically referring to different categories or types of problems AI is designed to solve. In this section, I’ll explain each workload clearly, and I’ve also referenced small demo videos or examples to help you understand how these workloads work in the real world.

The first workload is Machine Learning. Don’t worry if you don’t fully understand machine learning yet—we will deep dive into what it is, how it works, and the different types of machine learning later. For now, just remember that machine learning is a branch of artificial intelligence and computer science that uses data and algorithms to imitate the way humans learn and gradually improve accuracy. A simple example is how a child learns to tell the difference between a cat and a dog. When a child is born, they have no idea which is which. But over time, through books, pictures, and guidance from parents, the child learns the features of a cat versus the features of a dog. Machine learning works similarly: we teach a computer model to make predictions and draw conclusions from data, just like humans learn from experience.

A great real-world example is Netflix. When you watch movies on Netflix, the platform recommends new movies or TV shows based on your viewing history. If you watch a lot of thrillers, Netflix learns that and recommends more thrillers. If you prefer romantic comedies, it will show you romcom suggestions. Netflix uses machine learning techniques such as A/B testing to compare algorithms and find out which recommendations bring more viewer satisfaction. Their system continuously learns from user interactions to provide better suggestions.

The next workload is Computer Vision. This is an area of AI that enables computers to identify, detect, and classify objects within images or videos. Until recently, computers could store photos and videos, but they couldn’t actually understand what objects were present in them. With computer vision, machines can visually interpret the world using cameras, images, and video streams. For example, a system can look at a video and identify people, vehicles, objects, or even track movements. If there was a ball in the scene, the computer could detect that it is a ball. This workload forms the basis for technologies like self-driving cars, surveillance systems, medical imaging, and facial recognition.

Another major workload is Natural Language Processing (NLP). As the name suggests, natural language refers to the way humans communicate—whether through spoken or written language. NLP equips computers with the ability to understand human language and respond accordingly. A perfect example is Alexa. You simply talk to Alexa in English, Hindi, or any supported language, and it interprets your speech, processes your request, and responds naturally. Whether you are asking Alexa to solve math problems, make an announcement, or perform a task, you are interacting with an NLP-enabled system that understands your voice commands and replies just like a human would.

Next, we have Generative AI (GenAI), which is one of the biggest technological shifts today. Earlier, AI systems mainly focused on predictive tasks, where the outcome was already defined—for example, predicting whether a person has diabetes based on input data. You feed the data, and the system predicts a yes or no outcome. Generative AI, however, goes much further. Instead of only predicting, it has the ability to create original content. This content can be in the form of text, images, code, diagrams, videos, and much more.

For example, suppose you want to surprise your wife on her birthday by writing a poem but you’re not confident about your writing skills. With generative AI, you simply provide some details—where you met her, what she likes, her positive qualities—and the system will instantly generate a beautiful poem. Similarly, you can ask it to write Python code, create an image, produce an architectural diagram, or generate a detailed explanation. Tools like ChatGPT demonstrate this perfectly. When you ask ChatGPT to write a 400-word review on gas sensor datasets, it produces the content within seconds. The speed and creativity are remarkable, making generative AI a revolutionary leap forward in how we interact with technology.

With this overview, you should now have a clear understanding of the various AI workloads—Machine Learning, Computer Vision, Natural Language Processing, and Generative AI—and how each plays a major role in shaping the world of artificial intelligence.

# **E) AI vs ML vs DL**

Another key question that comes up when studying AI is understanding the difference between Artificial Intelligence, Machine Learning, and Deep Learning—often referred to as AI vs ML vs DL. The best way to visualize this is like an onion with multiple layers. Each layer represents a level of abstraction, with AI at the top, ML in the middle, and DL at the core.

At the topmost layer is Artificial Intelligence (AI). As we’ve discussed, AI is the capability of a computer system to mimic human-like cognitive functions, where cognitive stands for knowing, learning, and understanding. AI is the broadest field and encompasses any technique or system that enables machines to perform tasks that typically require human intelligence.

Beneath AI lies Machine Learning (ML). Machine learning is a subset of AI and a branch of computer science that focuses on using data and algorithms to imitate the way humans learn. ML systems improve their accuracy over time through experience. A simple analogy is a baby learning to distinguish between a cat and a dog. Initially, the baby cannot tell the difference, but through observation, guidance from parents, and books, the child learns to recognize the features of each. A modern example is Netflix, which uses machine learning to recommend movies and shows based on your preferences. The system learns from your interactions and improves its recommendations over time. The key idea here is learning and improving—machine learning systems get better as they process more data.

At the innermost layer is Deep Learning (DL). Deep learning is a subset of machine learning that teaches computers to process data in ways inspired by the human brain, particularly using neural networks. It enables the computation of multi-layer neural networks, making complex tasks feasible. While we will explore neural networks in more detail later, it’s important to understand that deep learning forms the foundation for advanced technologies like driverless cars and generative AI systems. Deep learning allows machines to process vast amounts of unstructured data, recognize patterns, and make complex decisions autonomously.

In summary, the relationship between AI, ML, and DL can be visualized as layers: AI is the topmost layer, encompassing all intelligent behavior; ML sits beneath AI, focusing on systems that learn and improve; and DL forms the core, enabling highly complex tasks through neural networks inspired by the human brain. Understanding this layered mechanism helps clarify the distinctions and connections between these three critical areas of artificial intelligence.

# **II) Machine Learning Foundations (For Absolute Beginners-Optional)**

# **A) Machine Learning - Real World Example**

Okay, so let's take a look at a real-world example of machine learning. So it's just people who are actually working on machine learning at Netflix day in and day out, and they'd be sharing their experiences. So for the moment, just try to have a look. Don't worry. We'll be going into the theoretical part of machine learning just after this video. At Netflix, we have over 120 million members. They span the whole globe across 190 different countries, as well as a diversity of titles, content, comedies, and dramas. Machine learning is deeply intertwined with all aspects of Netflix's business—how we compose our catalog of content, how we produce our content, how we encode it, how we stream it. We use machine learning to help marketing efforts and advertising efforts. Then we also use machine learning in our content acquisition effort. Okay, so this was just a quick snippet or a video on machine learning in the real world. In the next video, we'll start learning about some of the key terminologies of machine learning because once you get a good understanding of the key terminologies, then you can have a good hold of the machine learning fundamentals. Thanks for watching.

# **B) Machine Learning - Key Terminologies**

Hi folks. Welcome back. So now it's time to go into the theoretical definitions and the different terminologies for machine learning. Before we actually do a deep dive into machine learning, I thought it would be better to give you a good understanding of different terminologies. This is very important because whenever you are reading about machine learning, artificial intelligence, or when you're talking to AI professionals or data scientists, these are the kind of words they will keep on saying. If you don't have a good understanding of these words or terminologies, you'll really struggle in the field of machine learning or AI. So let's take a look at these fundamental concepts. The very first one is an algorithm. What is an algorithm? A general definition of an algorithm, when we studied math or physics, is that an algorithm always refers to a set of rules. 

In machine learning, an algorithm refers to a set of rules and statistical techniques. Now you might remember what statistics is. A clear definition of statistics is that it is the study and manipulation of data, including ways to gather, review, analyze, and draw conclusions from data. In other words, you have a lot of data, and from that data, you need to draw conclusions. So algorithm is a set of rules and statistical techniques. A good example of this is the decision tree algorithm. A perfect example of that is Gmail. When Gmail came, it never had a concept of filters. After some time, they came up with the concept of filters to classify emails as spam or not spam. They look at your emails and, based on a certain set of rules, they tell you which emails are spam and which are not. They do this based on features—now called features or artifacts in machine learning—such as who the sender is, frequency of certain words, or specific phrases like "you have won a lottery" or "there is money in your account." These features help the algorithm decide whether an email is spam or not. This is your algorithm. The next concept is a model. A model is what an algorithm creates. 

Always remember, a model is derived from an algorithm. A model is essentially the learned representation of data. It is a program that can find patterns and make decisions. For example, a dataset of house features and prices fed into a regression algorithm can create a model that predicts the price of new houses based on their features. If you feed new data to the model, it can predict price ranges for houses. So a model is an algorithm that has been trained on data, and it helps find patterns to make decisions. Now comes the concept of training. You always train your model, similar to training a child to make decisions. Training is the process of presenting data to a machine learning algorithm to create a model. The algorithm uses training data to learn. For example, a neural network shown thousands of pictures of cats and dogs can learn to identify whether a new image is a cat or a dog based on features like ears, eyes, and other patterns. Then, there's the concept of labels. A label is the output you want the model to predict. For example, in healthcare, the label could be whether a tumor is malignant or benign. Malignant means dangerous, while benign means it is not a major issue. To summarize, algorithm is a set of rules, a model is what an algorithm creates after being trained on data, training is the process of presenting data to an algorithm to create a model, and labels are the outputs the model is trying to predict. Thanks for watching.

# **C) What is Machine Learning ?**

Hello and welcome. So the time has finally come to talk about machine learning. Before we go into the details of what machine learning is, it’s important to understand how humans learn. Just think about it—when a baby is born, can the baby distinguish between a cat, a dog, and a horse? No. A baby learns by observing different features. For example, if an animal has long ears and a black nose, it might be a dog. If it has small size and blue eyes, it could be a cat. Humans learn primarily by examples, diagrams, and comparisons. Similarly, we train machines to learn using data examples.

For instance, if we feed thousands of labeled photographs of dogs, cats, and horses to a computer, it can identify patterns based on features like ear length, nose shape, and eye color. These patterns are used to train an algorithm, and once the model is built based on that algorithm, it can make predictions on new data. This model can then classify a new image as a cat, dog, or horse based on its features.

Theoretically, machine learning is a branch of artificial intelligence. As discussed earlier in the AI-ML-DL “onion” analogy, machine learning (ML) is a subset of artificial intelligence (AI). ML focuses on the use of data and algorithms. If I were to define the heart and soul of machine learning, it would be data. Machine learning helps machines imitate how humans learn, gradually improving their accuracy. Computers apply statistical learning techniques to automatically identify patterns in data. Statistics itself is the study and manipulation of data, including ways to gather, review, analyze, and draw conclusions from data.

Data contains patterns, and algorithms are used to find these patterns. Using our cat and dog example, features in the data are analyzed to find patterns, which are then used to train the algorithm. Once trained, a model is created. This model can recognize patterns in new data. Data is typically divided into training data, which is used to train the model, and testing data, which is used to evaluate its accuracy. The model is then used to make predictions on new, unseen data.

As machine learning progresses, increased data and experience improve the accuracy of the results, similar to how humans learn gradually. Machine learning algorithms improve as they process more data, updating their parameters to learn over time. There are three main approaches to machine learning: supervised learning, unsupervised learning, and reinforcement learning.

Supervised learning involves labeled data. For example, in the cat and dog dataset, features are labeled to indicate whether the animal is a cat or a dog. Unsupervised learning deals with unlabeled data and helps identify patterns or structure in such data. Reinforcement learning is based on rewards and penalties. Similar to how a teacher rewards a student for correct actions and penalizes mistakes, machines learn by maximizing rewards and minimizing penalties over time.

Data is the foundation of machine learning. Both the quality and quantity of data are critical. Garbage data will result in poor models, while high-quality data improves accuracy. Likewise, larger datasets help models better understand patterns and improve predictions. For example, feeding 100 rows of data will train a model to some extent, but 10,000 rows will give the model a much better understanding of the patterns.

Algorithms in machine learning are sets of rules, often based on statistical or mathematical techniques. They are tailored to specific types of data and learning tasks. As seen in real-world examples like Netflix, machine learning is applied across diverse fields, including healthcare, finance (e.g., fraud detection), autonomous vehicles, and tasks like prediction, classification, clustering, and deployment. Once trained, models are deployed in real-world applications to provide insights, automate tasks, or enhance decision-making.

In conclusion, machine learning is essentially about teaching machines to learn like humans. By feeding data, finding patterns, and building models, machines can make predictions and decisions. Data-centric machine learning relies heavily on the quality and quantity of data. Always remember—think of machine learning as training a baby: the better and more comprehensive the training data, the better the machine learns.

# **D) Types of Machine Learning**

Okay, so now we have got a very good understanding of what machine learning is. We have seen some real-world examples of machine learning, and we also discussed the workflow of machine learning. Now, it is important to understand the various types of machine learning algorithms. In this section, we will do a deep dive into each of these types.

First, we will learn about supervised machine learning. Supervised learning is where the model training is done using labeled data. Don’t worry—we will explore this in detail, and we will understand concepts like classification and regression.

Next, we will talk about unsupervised learning. Unsupervised learning is called “unsupervised” because it uses unlabeled data. In unsupervised learning, we have the concept of clustering. Each of these algorithms or models addresses different use cases, which we will also discuss in detail later.

Then comes reinforcement learning. As discussed earlier, reinforcement learning works based on the reward and punishment method. Just like a baby or a student learns by receiving rewards for correct actions and penalties for mistakes, reinforcement learning works on a feedback loop. The model takes actions in an environment, receives feedback and state updates, and gradually learns to make better decisions.

So, with this quick overview, we can summarize that there are mainly three types of machine learning algorithms: supervised learning, unsupervised learning, and reinforcement learning. Supervised learning includes classification and regression, unsupervised learning mainly focuses on clustering, and reinforcement learning relies on feedback from the environment.

In the next video, we will go deeper into supervised learning. See you in the video. Thanks for watching.

# **E) What is Supervised Machine Learning ?**

Hello and welcome. It's time to take a look into supervised machine learning. Whenever we learn a new subject, it helps to look at its root word. In this case, the root word is “supervisor.” A supervisor is a person in charge of a group, ensuring that work is done correctly, accurately, and according to the rules.

The term supervised machine learning comes from the fact that the algorithm is trained using labeled data. Labeled datasets are annotated with meaningful tags or labels that classify the outcome. For example, in our earlier cat and dog example, we gave features such as nose type, ear type, and ear length, and labeled the animal as either a cat or a dog. In this case, the labeled dataset acts as the supervisor—it guides the algorithm in learning the correct output.

Supervised learning involves training a model on a dataset that includes both input data and the corresponding correct output. A classic example is the Iris flower dataset, which contains various types of Iris flowers, such as Setosa, Versicolor, and Virginica. Each flower has features like sepal length, sepal width, petal length, and petal width. The input data consists of these features, while the output data is the flower’s class. By providing thousands of such examples, the model can learn patterns in the data and make accurate predictions for new inputs.

The training dataset must be labeled, meaning each example is paired with the correct answer. For instance, if you provide sepal and petal measurements along with the corresponding flower type, the model learns to map the input features to the correct class. This mapping allows the model to make predictions on unseen data.

There are different types of algorithms in supervised machine learning, including linear regression, logistic regression, support vector machines (SVMs), decision trees, and neural networks (the latter being explored further in deep learning). The training process is straightforward: the model learns a function that maps inputs to the desired outputs and makes predictions based on that function. For example, if you feed a model new data such as sepal length 5.9, sepal width 3.0, petal length 5.0, and petal width 1.8, it should predict that the flower is a Virginica.

One risk in supervised machine learning is overfitting, where the model performs very well on training data but gives inaccurate results for unseen data. Overfitting occurs when the model learns patterns that are too specific to the training set, reducing its ability to generalize.

Supervised machine learning has a wide range of applications. Examples include image recognition, where models identify cats and dogs in images; speech recognition, such as voice-based banking systems where the model identifies a user’s voice as a password; medical diagnosis, where models help identify diseases; spam detection, which classifies emails as spam or not spam; and stock price prediction, which forecasts continuous values.

In supervised learning, there are two main types: classification, which sorts items into categories, and regression, which predicts continuous values. For example, classification can label an email as spam or ham, while regression can predict a house price based on features. We will explore these types in more detail in the next videos, looking at the differences between classification and regression.

# **F) What is Classification in SML ?**

So there are two types of machine learning, or supervised machine learning per se. One is classification and the other is regression. As I always say, it’s important to look at the root word. In classification, the root word is classify. You may ask, what does classify mean? It means to arrange a group of people or things into classes or categories. It’s as simple as that. Essentially, you have many things, and you are just grouping them.

For example, consider I have a brinjal, a broccoli, a carrot, and some burgers. You want to categorize them. If you pass this through a classification model or machine learning model, it can classify that all the vegetables fall under the category “vegetables,” and the burgers fall under “groceries.” If a company wants to do analytics to see what customers are buying, and categorize how much is vegetables and how much is groceries, this is exactly what classification does.

In terms of a formal definition, classification involves assigning category labels to new observations based on past observations and their labels. In supervised machine learning, we have labeled data. The aim of classification is to assign category labels, and these labels are known when you input your data. Classification can be binary, meaning two classes, like determining whether an email is spam or not (non-spam is also called ham). A lot of emails are fed into the classifier, and based on features or algorithm logic, it identifies which ones are spam and which ones are ham, sending them to the appropriate folder.

Classification can also be multi-class, such as classifying types of fruits. There are multiple fruit categories, and the model identifies which category each fruit belongs to. Common algorithms used for classification include Decision Trees, Random Forests, Logistic Regression, Support Vector Machines, and Neural Networks. While you don’t need to fully understand each yet, it’s important to know their names.

Feature selection is crucial in classification. Choosing the right features or inputs determines the effectiveness of the model. For example, in the healthcare sector, detecting prostate cancer requires feeding MRI images to the model. The feature design determines whether the image contains a lesion or not. A lesion indicates a damaged area or possible cancer. The machine, using these features, classifies whether the image falls under the prostate cancer category or another category. This is far more efficient than manually analyzing thousands of images. Selecting the right features is critical because if features are poorly chosen, the model might miss identifying a patient with cancer.

As a supervised machine learning technique, classification requires labeled data. Each instance in the training data is tagged with the correct class. When test data is introduced, the model should predict the appropriate category based on what it has learned from training.

Another important concept is overfitting and underfitting. Overfitting occurs when a machine learning model performs very accurately on training data but fails to generalize to new or test data. It is considered undesirable. Balancing model complexity to prevent overfitting and underfitting is essential. Testing helps adjust and tweak the model so that its performance on new data mirrors its performance on training data.

Classification has many applications. Email spam detection is one example. Image recognition is another; for instance, analyzing MRI prostate images to detect cancer. It’s also used in medical diagnosis and sentiment analysis, and it can even assist in real-time decision-making. In banking, classification is used for fraud detection. For example, based on transaction features and patterns, a system can classify transactions as normal or potentially fraudulent.

To summarize, supervised machine learning classification is straightforward. It involves classifying or adding category labels—essentially arranging a group of people or things into classes or categories. By understanding features and carefully training models, classification can be applied to various sectors effectively, from healthcare to banking to everyday analytics.

# **G) What is Regression in SML ?**

Hi folks. Welcome back. In the previous video, we discussed supervised machine learning classification. Now, we will talk about regression. You might ask, what is regression? Regression is a statistical technique that relates a dependent variable to an independent variable. You might also ask, what is a dependent variable and what is an independent variable? Let me explain with a very simple example.

If we have an equation like 𝑦=2𝑥y=2x, something most of us learned in algebra, here 𝑥 x is the independent variable and 𝑦 ;𝑥 x. For example, if 𝑥 = 2 x=2, then 𝑦 = 4 y=4. 

y dependent? Because its value changes as we change y=6. The independent variable is plotted on the x-axis, and the dependent variable is plotted on the y-axis, usually in 
a scatter graph showing various data points. Regression analysis predicts a continuous output, which is the dependent variable. Essentially, we try to determine the value of 𝑦 y based on one or more predictors (independent variables). For instance, in house price prediction, the general assumption is that the larger the house, the higher the price. So if a house is 2500 square feet, its price would likely be in a high range. If your test data has a house of 1250 square feet, regression helps predict the dependent variable, i.e., the price, which might be around 220 (thousand or lakhs, depending on units). Keep in mind, this is just a prediction, not the actual price.

There are two types of regression: linear regression and multiple regression. Linear regression is the simplest form and assumes a linear relationship between the input and output variables. For example, predicting house price based on its size is a linear regression problem. In linear regression, there is only one independent variable.

In contrast, multiple regression involves two or more independent variables. For example, predicting a car's price could involve features such as age, mileage, brand, engine size, and more. By including multiple independent variables, we can make more accurate predictions.

The line of regression is the line used to predict the value of 𝑦 y for a given 𝑥 x. This can be thought of as a model representation, like 𝑦=2𝑥 y=2x in a simple example, but regression algorithms can compute this line based on training data.

Regression has many applications. In finance, it can be used for stock price prediction based on historical data. In medical diagnosis, it helps predict outcomes or measurements. In retail, regression can forecast sales based on historical trends and market behavior.

As with classification, regression models can also suffer from overfitting and underfitting. Overfitting occurs when a model performs very well on training data but poorly on unseen data, such as predicting future stock prices. A complex model may fit historical data perfectly but fail to generalize to new data. Therefore, it is essential to tune the model so it performs well on both training and test data. Most machine learning algorithms require this balance.

To clarify the difference between regression and classification: if you are predicting whether it will be hot or cold tomorrow, you are categorizing the outcome—this is classification. If you are predicting the exact temperature tomorrow, you are predicting a continuous output—this is regression. Regression analysis is all about predicting the value of a dependent variable based on one or more independent variables.

# **H) What is Unsupervised Machine Learning ?**

After gaining a solid understanding of supervised machine learning, it’s now time to explore unsupervised machine learning. In unsupervised learning, there are mainly two types: clustering and association.

The golden rule to remember is that supervised machine learning works on labeled data. For example, if you feed a lot of pictures of cats and dogs, you label each picture as either a cat or a dog. This labeled data acts as the supervisor, guiding the machine learning model. In contrast, unsupervised machine learning deals with unlabeled data. Here, you feed the algorithm data without providing any output labels. For instance, you insert a lot of pictures, but you don’t tell the machine which ones are cats or dogs. This is the core of unsupervised learning.

By definition, unsupervised learning involves analyzing and clustering unlabeled data sets. You might ask, what is clustering? Clustering is essentially grouping data. The algorithm groups similar data points together—forming clusters. For example, cluster one contains data points that are similar, and cluster two contains another type of data points. Clustering allows the discovery of hidden patterns in the data. The goal of unsupervised machine learning, like supervised learning, is to understand patterns and make predictions, but here it does so without guidance from labeled data.

Clustering is the most common unsupervised learning technique. It groups data points into clusters where items in the same cluster are more similar to each other than to items in other clusters. For instance, if you feed different pictures into the algorithm, it might place all the cats in one cluster and all the dogs in another. This technique is widely used in market segmentation, where customers are grouped based on purchasing behavior. Companies can then target specific clusters with tailored marketing strategies.

Another important unsupervised technique is association. Association identifies sets of items that frequently occur together. A perfect example is online learning platforms like Udemy. When you purchase an Oracle Cloud course, the platform may recommend three other courses frequently bought together by other users. This helps learners discover relevant courses and often comes with bundled discounts. Similarly, in retail, association rules help find products often bought together, optimizing store layouts or promoting cross-sales.

Anomaly detection is another important aspect of unsupervised learning. It is used to identify unusual patterns, such as detecting hackers intruding into a network or spotting fraudulent transactions. By examining transaction patterns, the algorithm can flag behaviors that deviate significantly from the majority of data, enabling fraud detection in banking or cybersecurity applications.

Unsupervised learning has broad applications, including bioinformatics, image recognition, speech recognition, and recommender systems. Platforms like Netflix and Spotify use unsupervised learning to group users with similar interests. For example, if a user frequently watches thriller movies or takes Oracle Cloud courses, the system recommends similar content without needing labeled data for each user.

However, unsupervised learning comes with challenges. Compared to supervised learning, it is more difficult because there is no labeled data. One of the biggest challenges is determining the correct number of clusters in a dataset without predefined categories. While simple examples like cats and dogs are straightforward, real-world datasets often have many varieties, making it complex to identify clusters accurately. Data scientists must carefully tune their models to handle these complexities.

In summary, unsupervised machine learning focuses on discovering patterns in unlabeled data. Its two main types are clustering, which groups similar data points, and association, which identifies items that frequently occur together. Despite the challenges, unsupervised learning is a powerful tool for understanding hidden patterns and making informed decisions.

# **I) What is Re-inforcement Machine Learning**

In the previous videos, we learned about supervised and unsupervised machine learning. Now, it’s time to explore the third type of machine learning, called reinforcement learning. As always, it helps to look at the root word. Reinforcement means to further strengthen or to give additional strength. In everyday language, we often hear about reinforcement in the context of the army—sending a reinforcement army to strengthen a deployed force. Similarly, in machine learning, reinforcement means strengthening a model or moving toward more accurate predictions. The ultimate goal is to achieve maximum accuracy, and this is achieved through a process of continuous feedback and improvement.

The core concept of reinforcement learning involves an AI agent. The agent learns to make decisions by performing actions and receiving feedback. Feedback comes in the form of rewards and penalties. You can think of it like a student in a classroom. If the student does well, they receive a reward. If they make a mistake, they may face a penalty. Similarly, the AI agent performs actions, observes the results, and adjusts its strategy to maximize rewards and minimize penalties.

In reinforcement learning, the agent interacts with its environment. At each time step, the agent receives the state of the environment, takes an action, and gets feedback in the form of a reward or penalty. For example, in a game of chess, the environment is the chessboard, the actions are the moves (like moving a king, queen, or knight), and the feedback is whether that move improves the agent’s position (reward) or worsens it (penalty). The agent is the heart of reinforcement learning, learning continuously from its interactions with the environment.

The learning process involves selecting the best action given a state to maximize cumulative rewards over time. Two critical concepts are exploration and exploitation. Exploration refers to trying new actions to discover potential strategies, while exploitation means leveraging existing knowledge to make the best decision based on what the agent already knows. Balancing exploration and exploitation is a key challenge, as relying too much on either can reduce the model’s effectiveness.

Reinforcement learning has several important applications. Game playing, such as chess or Go, is a classic example. Robotics and autonomous vehicles also rely heavily on reinforcement learning. For instance, companies like Waymo constantly improve their autonomous vehicle algorithms by learning from real-world driving experiences. Accidents or mistakes act as penalties, prompting the model to improve and adapt.

However, reinforcement learning comes with challenges. It requires large amounts of high-quality data, balancing exploration and exploitation, and handling environments with high variability or uncertainty. In autonomous driving, for example, unexpected events—such as a child running onto the road or a cyclist falling—create an unpredictable environment. The AI agent must learn to adapt safely and efficiently in such scenarios.

In summary, reinforcement learning focuses on strengthening models through trial, feedback, and improvement. The agent interacts with the environment, takes actions, and receives feedback in the form of rewards or penalties to optimize performance over time. It is a powerful approach for complex decision-making tasks where continuous learning and adaptation are critical.

# **J) What is a Jupyter Notebook**

This is a very important concept: Jupyter Notebooks. I want to give you a clear understanding of what a Jupyter Notebook is, because if you’re working in machine learning, data science, or artificial intelligence, a solid grasp of Jupyter Notebooks is essential.

A Jupyter Notebook provides an interactive environment for writing and running code in various programming languages, most notably Python. The beauty of a Jupyter Notebook is that you can write your code and see the output in the same interface.

If you look at a notebook, you’ll notice it is organized into cells. Each cell can contain code, and you can execute it independently. For example, you can run one code cell, see its output, and then move on to the next cell. This live code execution makes Jupyter Notebooks very convenient for experimenting and iterating on your code.

Although Jupyter supports around 40 programming languages (including R, Julia, and Scala), Python is the primary language used for data science and machine learning.

Another advantage of Jupyter Notebooks is the ability to include rich text elements. You can add markdown text, equations, images, and links, which allows you to document your code comprehensively alongside its explanations. This makes notebooks highly readable and useful for sharing and teaching.

Jupyter Notebooks also have excellent integration with data visualization libraries. For instance, Python’s matplotlib allows you to create static, animated, and interactive visualizations. Other libraries like Plotly and Bokeh can be used for interactive graphs. When you run a code cell, the output—whether text, table, or graph—appears directly below it, making the experience highly interactive.

You can also share your notebooks easily. They can be exported to different formats, such as HTML, PDF, or even slides, which makes them ideal for presentations and collaboration. This is one reason Jupyter Notebooks are widely popular in both academia and research. They are used extensively in universities and research projects for teaching programming, computational thinking, and data analysis.

Furthermore, Jupyter Notebooks integrate well with several data science tools, such as Anaconda. This integration allows you to perform tasks like data cleaning, statistical modeling, machine learning, and more, all within a unified environment. Even Oracle Machine Learning tools support Jupyter Notebooks, enabling seamless use in enterprise environments.

So, that’s why understanding Jupyter Notebooks is essential. In the next video, we’ll do a demo showing how to use Anaconda to run Jupyter Notebooks. From there, we’ll move into machine learning demos and start practical work.

# **K) Demo: Install Anaconda**

The time has finally come to learn something practically. Until now, we have been focusing a lot on the theoretical aspects of machine learning, but now it’s time to understand it in a practical manner. For this, it’s very important to have a good understanding of Anaconda. You might ask, what is Anaconda?

Anaconda is an open-source distribution of Python and R, designed specifically for data science. Its main purpose is to simplify package management and deployment. The package versions in Anaconda are managed by a system called Conda, which analyzes your current environment before executing an installation. This ensures that existing frameworks or packages are not disrupted. The Anaconda distribution comes with over 250 packages pre-installed, making it highly convenient for data science, machine learning, and AI work. Therefore, anyone looking to work in these fields should definitely understand what Anaconda is and how it works.

Getting started with Anaconda is simple. By going to Anaconda.com, you can select the free download option. Anaconda automatically detects your system and provides the correct installation file. There are installers available for Windows, Mac, and Linux. For Mac users, there are two options depending on the processor type, either Intel or M1/M2/M3. Anaconda is open-source, user-friendly, and trusted by many companies. Behind the scenes, it uses Conda to run all your code efficiently and manage the environment seamlessly.

One important part of Anaconda is the Anaconda Navigator, which is a graphical interface that allows you to manage, integrate, and run applications, packages, and environments without needing to use the command line. It comes pre-built with several important packages used for data exploration, transformation, visualization, machine learning, and more. These packages make it easy to perform a wide variety of tasks in a single environment.

Once Anaconda is installed, you can open the Anaconda Navigator, which allows you to launch applications such as Jupyter Notebooks. Jupyter Notebooks provide an interactive environment where you can write and run Python code. Each notebook consists of cells that can be executed independently. You write your code in a cell, run it, and the output appears directly below the cell. This live interaction makes it easier to experiment with code and immediately see results.

The beauty of Jupyter Notebook is that it also supports the integration of various Python libraries, enabling the creation of graphs, visualizations, and interactive outputs within the same environment. This makes it an essential tool for data science and machine learning tasks. By combining Anaconda and Jupyter Notebooks, you have a complete setup for writing, testing, and visualizing code in a highly interactive way.

In this video, we provided an overview of how to use Python in Jupyter Notebooks through Anaconda. In the next session, we will work with a real dataset, specifically the Iris dataset, to see machine learning in action. This will help bridge the gap between theory and practical application in a hands-on way.

# **L) Demo: Understanding the IRIS Dataset**

Now I’d like to introduce you to the Iris dataset. You might recall from one of our previous lectures on supervised machine learning, where we discussed labeled datasets. The Iris dataset is a classic example of a labeled dataset, and it has a long history in the field of machine learning. In fact, it was first used in R.A. Fisher’s 1936 paper, which shows just how foundational it is. The dataset is also available on the UCI Machine Learning Repository.

So what exactly is the Iris dataset? Essentially, it contains several features related to iris flowers. These features include sepal length in centimeters, sepal width, petal length, and petal width. The dataset also includes labels that indicate the species of the iris flower. The main species in the dataset are Iris setosa, Iris versicolor, and Iris virginica. There are 50 entries for each species, giving us a total of 150 rows. Because it is a labeled dataset, each row not only contains the features but also the corresponding species category, which makes it ideal for supervised learning tasks.

Our first step is to understand the dataset before we start building models. You can easily download the Iris dataset from the internet, or I will also provide it in the course resources as a CSV file named iris.csv. Once you have the file, you can begin exploring it using Python in a Jupyter Notebook. To start, we typically import the pandas library and read the CSV file into a variable. This allows us to view and manipulate the data easily. For example, using the head function, you can quickly see the top five rows of the dataset, which show the sepal length, sepal width, petal length, petal width, and the species of each flower. This is essentially your training data, which will be used to train your machine learning model.

Next, you might want to understand the overall structure of the dataset. By checking the shape of the data, you can see that there are 150 rows and six columns. This gives you a sense of the size and the number of features in the dataset. Sometimes, you may only be interested in specific columns. For instance, if you want to display just the ID and species columns for the first ten records, you can filter the dataset accordingly. This allows you to focus on only the relevant information while exploring the dataset.

This overview provides a solid understanding of the Iris dataset and sets the stage for practical machine learning. In the next video, we will take the Iris dataset and create our own machine learning model. We will train the model using this dataset and then make predictions to see how it performs. Things are about to get interesting, so keep watching.

# **M) Demo: Creating & Training your ML Model**

The Iris dataset is one of the most famous and foundational datasets in machine learning. It dates back to R.A. Fisher’s classic 1936 paper and is still widely used for teaching classification techniques. Now that we have studied supervised machine learning theoretically, this is the perfect time to start applying those concepts practically. In this exercise, we will load the Iris dataset, split it into features and labels, create a machine learning model, train it, and finally perform predictions using the trained model.

We begin by splitting the dataset into X and Y. As explained earlier, the X-axis (or the independent variables) contains all the features, while the Y-axis (or the dependent variable) contains the label we want to predict. In the Iris dataset, we drop the ID and species columns from the input features because species is our output label, and ID is irrelevant for prediction. This leaves us with four key features: sepal length, sepal width, petal length, and petal width, which form our X. The Y variable contains only the species column, which represents the type of Iris flower.

To verify this separation, we print X.head() and Y.head() separately. Printing the X values shows the four numeric feature columns, while printing Y shows the species names such as Iris-setosa, Iris-versicolor, and Iris-virginica. This simple step ensures that our data is correctly divided into inputs and outputs before we move into modeling.

Next, we create our machine learning model. We use the Logistic Regression algorithm from sklearn.linear_model. Logistic Regression is a classification algorithm and fits perfectly for this dataset. We initialize the model with default settings and store it in a variable, for example, ml_model. After initialization, we train the model using ml_model.fit(X, Y), where X contains the features and Y contains the corresponding class labels. This is where the model learns patterns from the training data.

Once the model is trained, we move on to testing its predictive ability. Testing is always done on new data (data the model hasn’t seen during training). We create a test input by manually specifying sepal length, sepal width, petal length, and petal width. This simulates a real-world scenario where we would provide measurements of a new flower and ask the model to classify it. We pass this data to the model using ml_model.predict() and store the result as predictions.

Finally, we print the model’s prediction. Using only the four numerical features, the trained model is now intelligent enough to tell us which species the new flower most likely belongs to. For example, it may correctly classify the flower as Iris-setosa. This end-to-end example perfectly demonstrates how machine learning can take raw numerical inputs and produce meaningful predictions based on learned patterns.

# **III) Deep Learning Foundations (For Absolute Beginners- Optional)**

# **A) What is Deep Learning ?**

Deep learning is an advanced branch of artificial intelligence and machine learning, and as the name suggests, it takes us much deeper into understanding how machines can learn. To grasp deep learning properly, it helps to first think about the human brain. Our brain is made up of millions of interconnected nerve cells called neurons. These neurons send signals throughout the body and enable us to perform everything from breathing and walking to speaking and thinking. The key idea behind deep learning is to mimic the way these biological neurons process information and communicate with each other.

In deep learning, this concept is implemented through artificial neural networks. These are computer algorithms inspired by the structure and functioning of the human brain. Just like real neurons are interconnected, artificial neurons—or nodes—are also connected across multiple layers. A human brain contains millions of connections, and similarly, complex deep learning models contain many layers of artificial neurons working together to process data and make decisions. The resemblance between a biological neural network and an artificial neural network is the foundational idea behind deep learning.

One of the major advantages of deep learning is that it reduces the need for manual feature extraction. In traditional machine learning, humans often identify which features of the data are important—for example, labeling images as cat or dog or specifying sepal length and sepal width in the Iris dataset. In deep learning, this dependency is significantly reduced. The neural network automatically learns the important features from raw data, combining feature extraction and classification into a single unified process. Although human involvement does not disappear completely, deep learning greatly minimizes how much manual intervention is needed.

Deep learning models consist of multiple layers, and this is exactly why the term “deep” is used—because data passes through many layers of transformations. Each layer learns increasingly complex patterns. These layers of interconnected nodes work together to recognize patterns that would be difficult to identify manually. This structure allows deep learning to excel in handling unstructured data, such as images, audio, and text—data that doesn’t follow a fixed format. For example, deep learning can recognize faces in images using Convolutional Neural Networks (CNNs) or understand human language through NLP models.

Because deep learning relies heavily on large neural networks, it requires massive amounts of labeled data to train. Unlike machine learning models that might work with thousands of rows, deep learning often needs millions of examples to achieve high accuracy. With such large datasets comes the need for high computational power. This is why GPUs (Graphical Processing Units) are commonly used. GPUs are excellent at processing large volumes of data in parallel, which makes them perfect for training deep learning models. Later, when we discuss AI infrastructure, we will dive deeper into GPUs and how they accelerate training.

Deep learning is now used in a wide variety of applications. These include autonomous driving systems, medical diagnosis, natural language processing, speech recognition, fraud detection, and much more. Much of modern AI—like chatbots, virtual assistants, and image-based search—relies heavily on deep learning models.

Finally, deep learning is supported by powerful tools and frameworks such as TensorFlow, PyTorch, and Keras. Keras, in particular, is a user-friendly deep learning library in Python that makes it easier to build and train neural networks. In upcoming lectures, we will explore Keras in detail with hands-on demonstrations to give you a deeper understanding of how deep learning models are built.

# **B) What is a Neural Network ?**

After understanding what deep learning is, the next important step is to explore what a neural network actually means. Before we dive deeper, let’s quickly recap what we learned earlier: deep learning is a machine learning technique that teaches computers to process information in a way inspired by the human brain. Our brain contains millions of interconnected neurons, and these neurons work together across many layers to help us perform everyday tasks—eating, drinking, thinking, deciding, and so on. These biological neurons form complex networks, and this very idea forms the heart of deep learning.

The core mechanism that powers deep learning is the neural network. In biology, neurons are the building blocks of the nervous system, but in artificial intelligence we work with artificial neurons. So you may ask: what exactly is a neural network? A neural network is simply a machine learning model designed to make decisions in a way that resembles how the human brain works. Inspired by the brain’s structure, scientists created artificial neurons and connected them together so that these networks can learn patterns, identify features, and make informed decisions just as biological neurons do.

A neural network is built from layers of interconnected nodes. Just like biological neurons are connected to one another, artificial neurons also form connections through which information flows. The structure of a typical neural network consists of three major components: the input layer, where the data enters; the output layer, where predictions are produced; and the hidden layers, which sit in between. These hidden layers can be many in number, and they are the reason neural networks become “deep,” giving rise to the term deep learning. These layers enable the network to learn complex relationships in data.

Each artificial neuron in a neural network performs a simple mathematical operation. It receives inputs, multiplies them with weights, adds a bias, and then applies a nonlinear function known as an activation function. This activation function decides whether the neuron should activate or not. To understand this better, imagine feeding an image of a circle to a neural network. Suppose the image is 28×28 pixels. That means we have 784 pixel values, and each pixel is sent to a corresponding neuron in the input layer. These neurons are connected to neurons in the next layer through channels, and each connection carries a weight—for example, 0.8 or 0.2. These weights determine the strength or importance of the signals being passed along.

The process where information moves from the input layer to the output layer is called forward propagation. The network processes the inputs layer by layer, and finally makes a prediction—perhaps it incorrectly predicts the circle as a square. When the prediction is wrong, the neural network adjusts itself using a process called backward propagation. During this process, the weights are recalibrated to reduce the error—for example, a weight of 0.8 might be updated to 0.6. This adjustment continues gradually until the network becomes accurate. This is how neural networks “learn by example,” just like humans learn from repeated observations.

Neural networks also use biases, which are extra parameters added to neurons to help them make better decisions. Together, weights and biases help the network determine the final output through the activation function. As the training progresses, these values keep updating, improving the model’s performance.

There are several types of neural networks, each used for different applications. Feedforward Neural Networks are the simplest form, where information moves only in one direction. Convolutional Neural Networks (CNNs) are widely used for image recognition tasks, such as identifying objects, faces, and scenes in images. Recurrent Neural Networks (RNNs) are commonly used for speech recognition and sequence-based tasks—this is how virtual assistants like Siri or Alexa recognize your voice and respond appropriately.

Neural networks also power technologies we use daily. Predictive text on smartphones uses neural networks to guess the next word you might type—for example, after writing “Hello, how are,” the model predicts “you.” Beyond these examples, neural networks play key roles in self-driving cars (such as Waymo), medical diagnosis, financial forecasting, stock market trading, and many other real-world applications. While the examples seem familiar across machine learning, deep learning, and neural networks, what improves significantly is the accuracy, power, and ability to extract complex patterns automatically.

Ultimately, the entire idea behind neural networks—and deep learning as a whole—comes back to two fundamental goals: identifying patterns and making predictions. That is what every model aims to do. With this, we complete our understanding of what a neural network is.

# **C) Deep Learning Models**

In this video, we’re going to look at some of the most important deep learning models. Along the way, I’ll also introduce a few key terminologies you should be familiar with when studying deep learning or artificial intelligence.

Let’s begin with a quick summary. In the previous lecture, we discussed how artificial neural networks work. When you have an image, it is divided into multiple pixels, and each pixel becomes an input to a neuron. These neurons pass information to the next layer through connected pathways, and each connection carries a weight. Along with weights, we also use biases. The network has an input layer and an output layer, and if the output accuracy is correct, the model performs well. If the accuracy is not correct, we use backward propagation to adjust the weights and improve the performance. That’s the basic idea behind training a neural network.

Now, let’s explore the different types of deep learning models.
The first model we come across is the Convolutional Neural Network (CNN). As I always say, look at the root word: “convolution.” In simple English, convolution means something with twists and turns, and mathematically it refers to combining two signals to form a third signal. You can think of it as a deep learning technique that moves through various loops or transformations. CNNs are ideal for image and video processing. For example, if you want to classify whether an image contains a cat or a dog, you would use a CNN. Social media platforms also heavily rely on CNNs for image classification tasks.

The next model is the Recurrent Neural Network (RNN). Again, look at the root word: “recurrent,” meaning something that repeats. RNNs are designed for sequential or time-dependent data. For instance, if you want to predict tomorrow’s stock price, you may use inputs such as yesterday’s and today’s prices. The network uses weights and biases, applies an activation function, and produces an output—in this case, the predicted stock price for the next day. RNNs also work well with natural language tasks, such as predicting the next word in a sentence. That’s how auto-complete on your smartphone works; when you type “How are,” it predicts “you?” automatically.

A special type of RNN is the Long Short-Term Memory Network (LSTM). LSTMs are excellent at handling long-range dependencies in sequences. A perfect example is Google Translate. When you type “Hello, how are you?” in English, the system identifies the language and translates it to another language—such as French—while maintaining context. LSTMs enable this capability, and they also support voice-based translation.

Next, we have GANs (Generative Adversarial Networks), which have become incredibly popular in recent years. GANs consist of two neural networks: the generator and the discriminator. These networks compete with each other. The generator creates fake images or videos, while the discriminator tries to distinguish between real and fake. The goal is for the generator to produce results so realistic that even the discriminator cannot detect the difference. This technique is behind deepfake videos—highly realistic but artificially generated footage of events that never actually happened. GANs make it possible to create photorealistic images and videos.

Finally, we come to one of the most important models in modern AI: the Transformer Model. This is the heart and soul of Generative AI (GenAI), which is extremely popular today. Transformers use a mechanism called self-attention to process sequential data more effectively than traditional RNNs or LSTMs. We will study transformers in detail in the next lecture, but for now, think of them as the foundation behind systems like OpenAI’s GPT-3. Models like ChatGPT use transformers to generate human-like text for applications such as chatbots, content creation, essay writing, poetry generation, and much more.

So this lecture was meant to build a basic understanding of the major deep learning models without diving too deep into the technical complexities. The key models to remember are:

CNNs for image and video processing

RNNs for sequential tasks like stock prediction or text auto-completion

LSTMs for long-range sequence learning, such as translation

GANs for generating realistic fake images and videos (deepfakes)

Transformers as the backbone of modern generative AI

With this, we come to the end of the lecture.

# **D) What is a Transformer Model?**

What if I tell you that what we are going to study today is the Transformer Model, which is the backbone of modern Generative AI? Everything you see today in the form of ChatGPT or any generative AI features offered by cloud vendors—at the heart and soul of all of it—is the transformer model. And this transformer model is a type of deep learning model that we should definitely be aware of.

So, what exactly is a transformer model?

As I always say, look at the root word. Whoever coined the term “transformer model,” why did they call it transformer? Think back to your physics classes. In electrical circuits, a transformer is a device that transfers electric energy from one circuit to another. You have Circuit C1 and Circuit C2, and the transformer simply transfers the electrical energy from C1 to C2.

Now, when we talk about transformers in neural networks, the idea is similar. Transformers are a type of neural network architecture that have revolutionized the field of Natural Language Processing (NLP). This is truly a game changer. And once you see the architecture diagram of a transformer, you can easily correlate how information is transferred from one stage to another—just like electrical energy is transferred between circuits.

So here is how the transformer architecture looks. You have an input layer, an output layer, a feed-forward network, and two important components: the Encoder and the Decoder. You may ask: what is an encoder? Encoders are neural network layers—usually multiple layers—that process the input sequence and produce a continuous representation of it. You can consider this as the embedding of the input.

Then we have the Decoder. The decoder uses these embeddings to generate the output. Don’t worry if this feels abstract right now; we will understand it with examples. For now, simply remember that transformers use a unique architecture based primarily on self-attention.

Now, what is attention? For the moment, think of “attention” as “context.” This model understands context—it knows what you really mean. Transformers use self-attention to weigh the importance of different parts of the input. In other words, it can look at the entire sequence of data and decide which parts are more relevant when generating the output.

And here comes the biggest advantage of transformers:
Unlike RNNs or LSTMs, transformers process the entire sequence in parallel.
This is why ChatGPT responds so fast. You simply ask it to write a poem, an email, or a story, and within seconds it generates it—because everything is processed simultaneously. It has already been trained on humongous amounts of data, and from that training, it draws inferences instantly.

Compare this with RNNs. In RNNs, the model processes data step-by-step. For example, if you type: “Hello, how are…”, the RNN looks at each word sequentially to predict “you?”. It’s more like the auto-correct or auto-complete feature on your phone. But transformers do not work like that—they handle entire sequences together.

Now let’s come back to the important concept we hinted at: Attention.

The core of the transformer is the attention mechanism. Even if you don’t understand the full architecture, if you understand attention, you understand the essence of transformers. Attention allows the model to focus on the right parts of the input while producing the output. And this enhances the model’s ability to understand context and relationships.

Let’s look at an example. Take the word “bank.”
In the sentence “the bank of the river”, “bank” refers to the side of the river.
In the sentence “money in the bank”, “bank” refers to a financial institution.
The word is the same but the meaning is completely different.
This is where attention shines. The transformer remembers context—so it understands the meaning based on surrounding words.

Now let’s take another example: writing a story.
If you ask a transformer model, “Write me a story,” 95% of the time stories begin with “Once upon a time.” But how does the model decide that?

This happens through a concept called softmax, which converts scores into probabilities. Suppose the model has options like “once,” “somewhere,” “there,” or completely unrelated words like “zygote.” It will give weightage to each option based on context. Since it knows you want a story, “once” becomes the most probable choice. Then it predicts the next word—“upon”—then “a,” then “time,” and so on.

Again, this entire process happens in parallel, not step by step.

As we continue, remember two key components of transformers:

Encoder

Decoder

Each encoder and decoder consists of multiple layers called transformer blocks, and every block contains:

Self-attention layer

Feed-forward neural network

Also remember, transformers do not need recurrent layers, and this parallelism makes them extremely scalable. That’s why GPUs—with thousands of cores running in parallel—are the preferred hardware for training transformer models.

Transformers have become the backbone of many state-of-the-art NLP models, such as:

Machine Translation (English → German, etc.)

Text Generation

Summarization

Question Answering

Two popular transformer-based architectures are:

BERT — Bidirectional Encoder Representations for understanding language

GPT — Generative Pre-trained Transformer

And yes, the T in GPT stands for Transformer, which is exactly why we are studying this model now.

Beyond NLP, transformers are also making their way into computer vision and audio processing—anywhere tasks can be parallelized and context matters.

And with this, we come to the end of our lecture on the transformer model.

# **E) Demo: GANs - Deep Fake Video**

In the previous video, we talked about GANs, where we discussed the two main components: the generator and the discriminator. I also mentioned that one of the most popular applications of GANs is the creation of deepfake videos. Now, you might wonder—what exactly is a deepfake video?

A deepfake video is essentially a type of artificial intelligence technique used to create highly convincing fake images, audio clips, or videos. The term “deepfake” refers to both the technology behind it and the final output, which is completely fabricated. It’s not real content, which is why we call it “fake,” and because it uses deep learning techniques, the word becomes “deepfake.” So, using ML and DL models, we are able to generate extremely realistic—but entirely artificial—media.

Let’s look at an example to understand this better. There was an AI developer who created a deepfake video of Tom Cruise. I’m sure all of you know who Tom Cruise is. This video went viral across social media platforms like TikTok and Instagram, attracting millions—even billions—of views. During a talk at TED, the host asked if they could have their own Tom Cruise deepfake video, and the developer actually shared one. In the clip, the AI-generated Tom Cruise says:

“What’s up, Internet? I’m north of the border, at the TED conference. It’s not short for Theodore, but nobody calls me Thomas, so it’s cool. It’s Tom at TED. Yes, I Canada. Seriously though, everybody here—very nice, very polite. Especially the waves.”

This short clip was a perfect demonstration of how realistic deepfake videos can be. Even though Tom Cruise never said those lines or attended that event, the AI-generated version looked and sounded almost identical to him.

Deepfake videos are typically created in one of two ways. The first method uses an original video of a target person, and the AI manipulates it to make them say or do things they never actually did. The second method involves swapping one person’s face onto another person’s body in a video. This process is commonly known as a face swap. Both techniques rely heavily on deep learning models to capture facial expressions, movements, and voice patterns, and then replicate them with stunning accuracy.

With this, I think you now have a clear understanding of what deepfake videos are, how they are created, and why GANs play such an important role in generating them.

# **F) Demo- Creating & Training Deep Learning Model**

Now it's time to walk through a quick demo based on the deep learning methodology we have learned so far. For this demonstration, we will use a dataset available on the Kaggle website. If you visit Kaggle, you will find the Pima Indians Diabetes Database. You may wonder: who were the Pima Indians? The Pima were a group of Native Americans living in the region that is now central and southern Arizona. Researchers collected medical data from this community and created a diabetes-focused dataset that is widely used in machine learning and deep learning experiments.

According to the description, this dataset originates from the National Institute of Diabetes and Digestive and Kidney Diseases. The main goal of the dataset is to predict—based on diagnostic measurements—whether a patient has diabetes. The dataset includes several health-related features for each patient and uses these measurements to determine whether the individual is diabetic or not. One important thing to note is that all the patients in this dataset are females and at least 21 years old, and all belong to Pima Indian heritage. This makes the dataset highly specific and controlled for certain demographic factors.

You can either download the data directly from Kaggle or use the copy found in the resources section. The dataset contains multiple columns, and the last column is extremely important: it contains the value 1 if the person has diabetes and 0 if they do not. As for the other columns, they represent various medical metrics. These include the number of pregnancies, plasma glucose concentration, blood pressure, triceps skinfold thickness, insulin levels, body mass index (BMI), diabetes pedigree function, and finally, the age of the patient. All of these factors together help determine whether a patient is likely to have diabetes.

With this understanding, we can now start building a deep learning model. We will create our own neural-network-based model and train it using this diabetes dataset. As always, it is best if you follow along. Start by opening Anaconda and launching a Jupyter Notebook. Once the notebook opens, create a new Python kernel and get ready to begin. Before we proceed, there are two very important terms you should know: Keras and TensorFlow. Keras is an open-source library that provides a Python interface for building artificial neural networks. It acts as a high-level API that makes deep learning easier to implement. TensorFlow, on the other hand, is a powerful open-source library developed by Google for machine learning and artificial intelligence, with a strong focus on training and deploying deep neural networks. Keras actually runs on top of TensorFlow, which is why we use the two together in this demo.

The first step is to import all the necessary Python libraries. Simply copy the import commands and paste them into your Jupyter Notebook. Once you run the cell, your environment will be ready. Next, we load the dataset. Here, you provide the path to your local file—this path will differ depending on your system—and specify that the file is comma-delimited. After loading the data, we must split it into the input features (X) and the output variable (Y). The X values correspond to columns 0 through 7, representing all the predictor variables. The Y value is simply the final column, which indicates whether the individual has diabetes.

Once the data is ready, we define our Keras model. This involves creating a sequential model and specifying the layers of our neural network. After defining the model, we compile it. For compilation, we use the Adam optimizer and track accuracy as our performance metric. Compilation prepares the model for training. The next step is to train the model using the model.fit() command, where we pass in the X values, Y values, and specify the batch size. When you run this cell, the model begins learning from the data.

After training, we evaluate the accuracy of the model. When we run the evaluation step, we find that the model is approximately 77% accurate. If we train the model again, the accuracy may improve slightly—this is a common behavior in deep learning. In this case, the accuracy increases from about 77.21% to 77.34%. This demonstrates why the size and quality of the dataset are so important. Larger datasets generally allow the model to learn more patterns and achieve better accuracy.

Next, we make predictions using the trained model. After generating the predictions, we compare the first 50 results to the actual expected outcomes. In many cases, the model correctly identifies whether the person is diabetic or not. However, there are also cases where the model fails. For example, the model may predict a 0 (non-diabetic) while the expected result is 1. These errors occur because the dataset is relatively small and the model has limited information to learn from. With a larger dataset, the accuracy would likely increase.

Through this demo, you can clearly see how deep learning works in practice. You have already studied the theory earlier, and now you can connect it with the hands-on implementation using Python, Keras, and TensorFlow. You’ve seen how data is prepared, how the model is built, trained, evaluated, and then used for predictions. I hope this practical walkthrough has helped strengthen your understanding of deep learning.

# **IV) Generative AI Foundations (For Beginners)**

# **A) What is Generative AI ?**

So if you ask me today which is the hottest and most talked-about technology in the IT world, I would say without any doubt it is Generative AI. Now you might immediately ask, “But what exactly is Generative AI?”—and that's a great question.

Before we dive into the meaning of Generative AI, I want you to recall the onion structure we discussed in our very first lesson. At the outermost layer, we had Artificial Intelligence. Then inside that, we had Machine Learning, which is a subset of AI. As we peeled further, we understood Deep Learning, which uses neural networks and forms a subset of Machine Learning. And now, inside that deep learning layer, sits Generative AI. So remember this important point:
Generative AI is a subset of Deep Learning.
Whatever concepts you learned in Deep Learning—especially neural networks—those form the backbone of Generative AI. This is why a strong understanding of Deep Learning always helps when you move into Generative AI.

So what is Generative AI? As always, I tell you to focus on the root word. The root word here is generate. Generate means to create, to produce, to build something new. So Generative AI is a type of AI that can actually create content—something new that did not exist before. And this content can be anything: text, images, music, videos, and even code. Yes, you can even ask a Generative AI system to write a Python program for you, whether it is a simple script or a complex piece of code.

Generative AI systems learn from extremely large datasets. So again, two things matter a lot here—quality and quantity. A large amount of good-quality data is needed because these models learn patterns, styles, structures and then generate new content based on what they have studied. This ability to understand complex patterns is what allows them to produce such high-quality outputs.

As mentioned earlier, the backbone of Generative AI is Deep Learning, especially neural networks. Most modern generative models, including the ones behind tools like ChatGPT, rely on deep neural networks trained on huge amounts of data. These networks work behind the scenes to generate responses, images, music, or whatever content you request.

Now let’s talk about the applications. Generative AI is used widely in art, where it can create stunning new images, paintings, or designs. In entertainment, it can create videos and animations. In marketing, companies use it to create ad content, taglines, and visuals. And in software development, developers can ask it to write entire code snippets or even debug existing code. So the possibilities are massive.

One of the biggest advantages of Generative AI is efficiency. Earlier, content creators had to write, rewrite, edit, and polish everything manually. Now, with generative systems, you simply provide a prompt or input, and within seconds, you receive an entire article, poem, presentation, or design. The creation process gets automated, reducing hours or even days of effort into minutes.

However, like any powerful technology, Generative AI comes with challenges, especially around copyright, originality, and misuse. Anyone can generate content and claim it as their own, which raises ethical concerns. So while using AI-generated material, we must be aware of these issues and use the technology responsibly.

Generative AI continues to evolve at a rapid pace. Every week, you will hear about new models being introduced—faster, smarter, and more accurate than before. These systems start from a simple user prompt and expand it into a full piece of content. For example, if you say, “Write a poem about my wife, whom I met in London,” and provide a few personal details, the AI will instantly produce a complete, heartfelt poem as if written by a poet. This is the power of user-prompted generation.

Finally, Generative AI is impacting multiple disciplines—journalism, design, education, marketing, programming—and transforming how work is done. But again, understanding its impact and limitations is equally important.

So for now, the key takeaway is:
Generative AI = a subset of Deep Learning that focuses on creating new content.

In the next video, we will explore the difference between Predictive AI and Generative AI.

# **B) Predictive AI vs Generative AI**

Okay, so the million-dollar question that is always asked is:
What is the difference between Predictive AI and Generative AI?

Now what if I tell you that everything we have learned so far—whether it was Artificial Intelligence, Machine Learning, or Deep Learning—most of those topics were actually dealing with patterns. And based on these patterns, I told you repeatedly: What are you looking for?
Yes—Predictions.

So all of that falls under Predictive AI.
But then what exactly is Generative AI?

As I told you in the previous lecture, the root word in Generative is generate, which means to create. So always remember this one golden sentence:
In Generative AI, you are creating new content.

Now let’s take a deeper look.

The function of Predictive AI is very clear—it is designed to analyze the existing data and make predictions. The root word here is predict. You are predicting future events or outcomes based on the data that you have trained the model on. And like I always say, when we talk about data, two things matter the most—quality and quantity. The better the data, the better the predictions.

Now let us talk about Generative AI.
Generative AI focuses on creating new content. It creates new data that resembles the training data, and sometimes even produces completely original output. This could be text, this could be images, videos, music, or even code. So yes, you can ask a Generative AI system to write a Python program for you as well.

When we look at usage, Predictive AI is mainly used for data analysis. It often involves statistical methods and machine learning algorithms to identify patterns and trends in historical data. For example, when you studied image classification—distinguishing between a cat and a dog—or when we talked about the Iris flower dataset where we predicted whether a flower is Iris Setosa, Iris Virginica, or Iris Versicolor based on features. All of that is Predictive AI.

We also discussed stock market prediction, where models try to predict the future price of a stock. Again—based completely on historical data patterns.

But things change when we enter the world of Generative AI.
Here, the focus is on content creation. Generative AI can generate text, images, music, synthetic data, simulations—anything that involves creating new output. You give a prompt, and it creates something for you.

Let’s talk about applications.
Predictive AI is used in areas like weather forecasting, stock market prediction, market price forecasting, risk assessment, customer behavior prediction, and fraud detection. These are all cases where we want to know “What will happen next?”

Generative AI, on the other hand, is used in art, design, media creation, synthetic dataset generation, and creating realistic simulations. Anywhere new content is being created, that is where Generative AI shines.

Predictive AI also helps in providing insights, recommendations, and better decision-making. For example, we saw recommender systems—like Netflix suggesting movies or Udemy suggesting related courses. These systems analyze your data and predict what you might like next. This is Predictive AI.

But Generative AI has the ability to produce creative and novel outputs that go beyond the input data. If you want a poem, if you want a letter, if you want an essay—just provide a few words, and Generative AI expands them into beautiful, full-length content. That is the magic of generative models.

Now a very important conceptual point:
Predictive AI is reactive in nature, while
Generative AI is proactive in nature.

Predictive AI reacts based on existing data and tells you what is likely to happen next. It is only analyzing.
Generative AI proactively creates new data points, new content, new ideas. You are not just analyzing anymore—you are generating.

For more examples, Predictive AI includes credit scoring models, demand forecasting, supply chain prediction, and predictive maintenance.
Generative AI includes ChatGPT and GPT models for text, DALL·E for images, AI-based music composition, synthetic data generation, and AI code generation.

With this lecture, I believe you now have a very strong and clear understanding of the differences between Predictive AI and Generative AI.

# **C) What is LLM ?**

Okay, so one of the most important concepts that we study in Generative AI is something called LM. Now everywhere you go, you will hear the word “LM.” But what exactly is LM?

LM stands for Large Language Models.
As the name suggests, these models are really large. They are not small machine learning models that you may have seen earlier. Here we are talking about models with multi-millions or even billions of parameters. That is why they are called large and why they are so powerful.

And since the word “language” appears in the name, that tells you they are related to NLP—Natural Language Processing, which we studied earlier. So what are LLMs?
LLMs are sophisticated AI systems designed to process, understand, and generate human language.

This means that whenever you give an input—for example, “Write a poem” or “Create a short story”—the LM is able to understand your request, process it, and then generate human-like text based on that. That is why LLMs are at the heart of modern Generative AI.

Now let’s talk about the most important aspect: scale and complexity.
The word large is not just for show. As the number of parameters increases, the complexity of the model also increases. When I say parameter, don’t worry—we will go deeper into what parameters actually mean. For now, just understand that LLMs are composed of millions or billions of these parameters, which is what makes them capable of producing such intelligent and human-like responses.

The best example is GPT-4 by OpenAI, which has an extremely large number of parameters and can generate very realistic and coherent text.

But now the question comes: What are parameters?
Let me give you a simple analogy. Imagine you are building a house. The house will have different parameters—like the number of rooms, the size of each room, the layout, the design, the type of flooring, the color of the walls. All of these are parameters of your house.

Now translate this to a large language model. Its parameters relate to what it can do. For example, the model may have the ability to generate text, translate languages, summarize articles, write poems, answer questions, create stories, or write letters. These are only small examples—LLMs have billions of such internal parameters that allow them to perform a huge range of tasks. This is why they become so complex.

LLMs are trained on massive datasets—far beyond normal sample data you might use in machine learning. They are trained on a wide range of internet text. For example, GPT-4 is trained on huge collections of books, articles, websites, and other diverse text sources. This vast training data helps the model understand grammar, context, reasoning, and even creativity.

Now let’s look at some popular LLMs. The list keeps expanding, but some of the major ones are:

– From OpenAI: GPT-3, GPT-4

– From Google: Bard (earlier) and now Gemini, which is becoming very popular and may soon be available on mobile devices

– Cohere, which is widely used in Oracle Cloud (OCI)

– LLaMA from Meta (Facebook), which is open-source and heavily used in research

These are some of the leading large language models in the industry today.

Now why do we use LLMs?
Because they help in natural language understanding and natural language generation. That means they understand human language and can respond in the same style. They can write essays, poems, simulate conversations, answer questions, and create human-like text in many forms.

The applications are huge. LLMs can be used to build chatbots, create content, translate languages, summarize documents, analyze text sentiment, and much more. For example, GPT-4 currently powers some of the most advanced chatbots used around the world.

Because they are built using machine learning—more specifically deep learning—these models are constantly improving. Remember, with deep learning we studied neural networks, which mimic how the human brain works. LLMs use a special type of neural network architecture called transformers, which we already studied.

This architecture allows them to process sequences of data efficiently. GPT-4, for example, is based on transformer architecture, which makes it great at understanding long paragraphs, context, and relationships between words.

A very important point is that LLMs are designed for continuous learning and improvement. They keep improving over time as they get more feedback, more usage, and more diverse prompts from users all around the world. That is why their performance gets better over time.

Another advantage is customization. You can use an existing pre-trained model, or you can customize (fine-tune) it for industry-specific tasks. For example, GPT-4 can be fine-tuned for legal language, medical reports, financial analysis, and many specialized domains. This is very useful when you want highly accurate output for a specific field.

Now if we look at this simple architecture diagram, we can see different types of inputs such as text, structured data, or even voice. These inputs go to the training model. The large language model adapts using the training data, learns continuously, and generates outputs. These outputs can include instructions, information, object recognition, image captioning, Q&A, quizzes, or even sentiment analysis.

For example, when we study OCI language services, we will see how LLMs perform tasks like identifying whether a text expresses positive, negative, or neutral sentiment.

So by now, I believe you have a very clear understanding of what an LM—or Large Language Model—is, how it works, what it can do, and why it is such an important part of Generative AI.

# **D) What is Embedding ?**

Embeddings are a really important concept in generative AI and are considered the backbone of it, especially when you study systems involving vectors, vector databases, or concepts like RAG. Embedding is essentially a way of converting the features of an object, such as a word, into a vector of real numbers, and while this is a theoretical definition, the main idea is that you convert the features of an object into numbers because vectors are nothing but numbers. To better understand this, consider a diagram using the example of a cat. A cat has several features, and we can assign numbers to each feature, such as whether it is a living being, belongs to a particular family like tigers or lions, whether it is human, its gender, whether it represents royalty, whether it is a verb, or whether it is plural. For each of these features, an embedding algorithm assigns specific numeric values — for example, the feature “living being” may get 0.6, while something like “royalty” may get a negative number that indicates it's not royalty. 

Similarly, words like houses get a positive value under plural, and the word kitten gets values very close to cat because both share similar features. These features form a seven-dimensional representation since there are seven features, but working in high dimensions is difficult, so we apply dimensionality reduction, where these embeddings are reduced from 7D to 2D. Dimensionality reduction transforms complex high-dimensional data into a simpler lower-dimensional space for easier visualization while still preserving as much significant information as possible. After reduction, we can visualize the embeddings in a 2D space; words with similar meanings or related relationships will appear close together—like cat and kitten—while unrelated words like dog or houses appear farther away. You can also observe relationships such as man-woman or king-queen, where the algorithm captures similarities in features like being a living being, being human, or having a certain gender. This demonstrates how generative AI systems understand context, meaning the system recognizes both meaning and relationships between words. The process is: start with a word, convert it into a word embedding (a numerical vector), reduce it from a high-dimensional representation to 2D, and visualize it in 2D where similar or related items cluster together. Embeddings transform complex data such as words or images into numerical vectors that computers can understand because computers work with numbers. 

These vectors help the system understand similarities—for example, cat and kitten are closer than cat and house. They also simplify complex data by breaking many features into smaller, easier-to-process numbers. Embeddings are used everywhere, such as in language understanding and recommender systems, and popular embedding methods include Word2Vec by Google, GloVe (Global Vectors), and BERT (Bidirectional Encoder Representations from Transformers). Computers learn to create embeddings by training on large amounts of data, and better data leads to better embeddings. 

Advanced embeddings can also understand context, such as distinguishing between the word “bank” used as a riverbank and “bank” used for money. For example, if you were to place an apple in a 2D embedding space, it would fall near other fruits like strawberry and banana based on shared features, just like sports items like football and basketball cluster together, or vehicles with wheels like bicycles and cars cluster in their own category. Overall, embedding is simply a way of converting the features of an object—like a word—into a vector of real numbers so that machines can understand meaning, similarity, and context. In the next step, you can learn about vectors in more detail.

# **E) What is a Vector Database ?**

Okay, so in the previous video we understood what embeddings are. Now it’s time to take a look at vectors and vector databases. A vector database is a type of database designed specifically for storing, indexing, and querying vectors, which are arrays of numbers representing data in a high-dimensional space. If we recall what we learned about embeddings, when we take a word like “cat,” we extract its features and assign numbers to each feature. These sets of numbers form what we call vectors. Essentially, all we are doing next is storing these vectors—these arrays of numbers or embeddings—into a database, and that database is known as a vector database.

You can imagine this by comparing it to relational databases like Oracle or MS SQL Server, where you store tables of relational data. In contrast, vector databases store vectors or embeddings. When you look at the diagram, you can see that different kinds of inputs can be processed. The input could be a document, a picture, a sound file, or even a video. These inputs are passed through different transformers: an NLP transformer for text, an audio transformer for sound, and so on. The transformer generates a set of numbers—embeddings—and these vectors are then stored inside the vector database.

Once data is stored, you can query it, index it, and perform searches, just like you would with any other database. The purpose is straightforward: vector databases are primarily used for storing embeddings that represent complex data like images, text, or audio in a numerical form that machines can understand and process. This is exactly what we discussed earlier—embeddings exist to break down complex objects into smaller numerical representations. For example, when we studied the example of the cat with seven dimensions being converted into a two-dimensional space, we saw how high-dimensional information can be reduced while keeping significant meaning.

These vectors are stored so we can perform similarity searches. Examples include “Is a cat similar to a kitten?” or “How close are they positioned in vector space?” Similarly, “Is a man related to a woman?” or “Is a king related to a queen?” These kinds of comparisons are done through similarity search mechanisms that vector databases are optimized to perform. They allow quick retrieval of items most similar to a given query. Some common similarity measures include Euclidean distance and cosine similarity, both of which look at how close vectors are in their multi-dimensional space.

Vector databases are widely used in machine learning, artificial intelligence, and recommendation systems. Think about Netflix—recommendation systems depend heavily on similarity search, such as finding what kind of movies you like based on prior viewing. Vector databases help handle very large volumes of data because the data, once converted into embeddings, becomes compact, structured, and easier for machines to work with. They support scalability, efficient querying, and are designed for big data applications.

Another important thing is that vector databases can be integrated with machine learning models. This allows them to directly store and query embeddings generated by these models. Most modern vector databases are built to work seamlessly with ML pipelines. They provide high-performance, real-time search and retrieval, which is essential for interactive applications and services. Just like traditional databases let you query tables, vector databases let you query vectors, but the purpose here is specifically to perform similarity searches.

So with this, we come to the end of understanding vector databases. In the next video, we will explore the relationship between embeddings, vectors, and vector databases. Thanks for watching.

# **F) Embedding vs Vector Database**

A lot of times when we introduce concepts like embeddings, vectors, and vector databases, people get confused because they aren’t able to clearly understand the differences between an embedding, a vector, and a vector database. That is why it becomes important to break them down and see how they are actually connected to each other. The best way to understand this is to always begin with the real world data. Real world data can be in the form of text, images, videos, or sound. All of these represent raw inputs that we want a machine to understand.

As we studied earlier, embedding is the process that transforms real world data into numerical numbers or vectors. These embeddings capture the important features of the data. For example, in the cat–kitten example, we took real world data and used embeddings to create numerical vectors. Embedding is the step that converts raw, complex information into usable numerical representations that a machine can process.

These vectors represent the features of the data. Vectors are nothing but the numerical representation of data created through the embedding process. Each vector is a list or array of numbers that encodes the characteristics of the data. When we saw the example of the cat—its category, whether it is feline, male or female, etc.—each of those features became part of the vector. From a high-dimensional representation like seven dimensions, we were able to reduce it to two dimensions, but the core idea remains the same: vectors are the numerical feature-based representation of the object.

Once you have these vectors, you need to store them somewhere. Just like you store your regular relational tables in Oracle or SQL Server, you store vectors in a vector database. A vector database is a special kind of database designed to store and efficiently retrieve large collections of vectors. It uses optimized techniques to quickly find vectors that are similar to each other. This is what enables similarity searches—like identifying that a man is related to a woman, a king to a queen, or a cat to a kitten.

We also discussed how vectors help group similar items, like fruits. If you create a vector for an apple, it will naturally fall close to other fruits because of shared characteristics. The vector database simply stores these vectors and retrieves them efficiently when needed. The flow is simple: real world data is transformed through embeddings to create numerical vectors, and these vectors (which represent the features of the data) are then stored in a vector database.

Finally, all of this happens in the backend, but the user interacts with it through an application at the frontend. Vector databases are used in various applications, especially similarity search, where we look for data points that resemble one another. A common example is Netflix, where the recommender system finds movies or shows similar to what you like. The same principle applies to many other recommendation systems. With this, you now have a clear understanding of the relationship between embeddings, vectors, and vector databases.

# **G) What is Retriever Augmented Generation (RAG) ?**

Hello folks, today we are going to study a very important concept within AI and generative AI called RAG, which stands for Retriever-Augmented Generation. At first, the term might sound a bit complex, but as I always suggest, it helps to break the word down and understand the root meaning. Whoever coined the term clearly had a logical idea behind it. The word retriever suggests fetching something, generation refers to creating something, and augment means adding value. So, RAG essentially means retrieving information, augmenting it with additional intelligence, and then generating a meaningful response.

To understand this, consider a simple example from the non-AI world. Suppose you are doing a PhD and you need to write a research paper. What would you do? You would go to a library, browse the internet, and gather information from various books and sources. These sources act as your knowledge base. Once you collect the information, you combine it all and write your paper. RAG works in a similar way. A user asks a question, the system looks into its knowledge base—which in our context is often a vector database—and retrieves the most relevant information. This is the retrieval stage.

Once the relevant information is retrieved, that retrieved text is combined with the user’s query. Together, they form a complete prompt or input. This prompt is then passed to a large language model such as ChatGPT or Gemini. The LLM uses both the user’s question and the retrieved content to generate a high-quality response. So in RAG, you are using the best of both worlds: your organization’s internal data as the source of truth and the power of the LLM to produce a refined and contextually accurate output. That is the essence of Retriever-Augmented Generation.

A natural question is: where do we actually use RAG? Think of a bank or any enterprise company. They have private, sensitive data stored inside their internal infrastructure. They do not want external LLMs like ChatGPT to directly access their confidential information. This is where vector databases step in. The retrieval happens from internal, secure data sources, while only the generation is outsourced to the LLM. In this way, RAG allows organizations to maintain full control over their data while still benefiting from generative AI capabilities.

In simple theory, RAG uses a retriever to find the relevant information and a generator to create the response. For example, if you ask, “What is climate change?”, RAG retrieves relevant scientific articles stored in your internal knowledge base and then uses the LLM to create a clear, concise explanation. This significantly improves response quality. Earlier, developers often had to hard-code responses or manually prepare structured rules. But with RAG, the system dynamically searches your source data in real time and generates accurate, detailed answers.

Another example is a customer support chatbot that uses RAG to fetch product details from an internal database. This ensures the responses are precise and fully aligned with company-specific information. RAG is highly adaptable across domains—customer service, education, enterprise chatbots, and any application where factual accuracy and context are essential. Organizations often cannot rely solely on the LLM’s built-in knowledge, so RAG ensures responses remain grounded in their own internal documents.

RAG also enhances contextual understanding. Throughout our discussions, we’ve emphasized the importance of context. With RAG, the context is retrieved directly from enterprise data sources, and the LLM interprets that context correctly. This helps avoid mistakes that purely generative models sometimes make, such as mixing up meanings—like “bank” as a financial institution vs. “bank” of a river. By grounding the generation in real documents, RAG ensures responses are accurate and contextually appropriate.

Let’s understand how RAG works in practice. First, you have your data sources: documents, PDFs, images, videos, and so on. All these are passed through an embedding model, which converts them into numerical vectors. These vectors are then stored in a vector database. This covers steps one and two: embedding the data and storing it in the vector DB. Next comes the user query. When the user asks a question in the chatbot, the query is also passed through the same embedder to generate a vector. This allows the system to perform a semantic search, not a simple SQL-style search. Instead of matching exact keywords, semantic search finds documents that are conceptually similar.

Once the relevant documents are found, they are sent to the LLM along with the user’s question. The LLM uses both pieces of information—the retrieved documents and the original query—to generate a final response. This combined context becomes the prompt for the LLM. The response is then returned to the user. So the workflow is: user submits a query → query is converted into a vector → similar documents are retrieved → both query and documents are processed by the LLM → final answer is generated.

This is the complete process behind Retriever-Augmented Generation. With this understanding, you can confidently explain RAG to anyone and describe exactly why it is so important for organizations today.

# **H) What is Langchian ?**

Today, we are going to understand a key concept in the world of generative AI called LangChain. To make it easier to grasp, think of LangChain like Lego. Most of us or our children have played with Lego. Lego consists of small bricks and comes with predefined instructions to build specific models. You can follow these instructions to build the model exactly as shown, or you can use the same bricks to create something entirely original based on your imagination. Similarly, LangChain allows you to use pre-built modules or models for common tasks or combine them in new ways to create complex language applications tailored to specific needs.

Just as Lego bricks are assembled following structured guides to create a final model, LangChain integrates various data sources with AI language models. For example, you might want to combine multiple LLMs such as LLaMA and ChatGPT. LangChain makes this integration possible. Essentially, LangChain is a technology framework that enables developers to build applications with large language models efficiently. You don’t need to start from scratch; instead, you can use pre-built modules, customize them, and assemble them as per your requirements.

A practical example of LangChain in action is the implementation of RAG (Retriever-Augmented Generation). Suppose you have an external data source, which is converted into embeddings and stored in a vector database. When a user query comes in, the same vector database is used to retrieve relevant information. This retrieved data acts as a prompt that is then sent to the language model. So, by using LangChain, you can easily implement RAG workflows. LangChain is modular and flexible, allowing developers to mix and match modules, which simplifies incorporating sophisticated AI functionality into applications.

LangChain is also very developer-friendly. Without it, integrating LLMs into software applications could become highly complex. The framework reduces implementation difficulty and enables developers to focus on building functionality rather than managing the underlying complexity of AI-driven conversational agents.

Here’s a simple example to make it concrete: imagine a user asks, “What is the weather like in Chandigarh today?” through an app integrated with LangChain. ChatGPT alone cannot provide current weather information since its knowledge is limited to its training data (e.g., GPT-4 is trained up to December 2023). LangChain processes the query, understands that it requires current weather data, and retrieves the information from the appropriate external source or API. Suppose the weather database returns sunny, 40°C.

Next, the language model comes into play. LangChain sends the retrieved data as a prompt to the LLM, which generates a natural, user-friendly response, such as: “The weather in Chandigarh today is sunny with a temperature of 40°C.” This response is then delivered to the user through the app. In this way, LangChain orchestrates the workflow: the query is processed, data is retrieved from the appropriate source, and the language model generates the response in a clear format.

In summary, LangChain is a technology framework designed to integrate large language models into applications effectively. It allows developers to use pre-built modules, customize workflows, combine multiple LLMs, and retrieve and generate responses using external or private data sources. LangChain makes it easier to implement complex AI-driven applications, such as RAG-based systems, while remaining flexible, modular, and developer-friendly.

# **I) Role of Langchain in RAG**

By now, we have understood the concept of RAG (Retriever-Augmented Generation) and have also touched upon LangChain. At this point, it’s common to feel a bit confused between RAG and LangChain. To clarify, LangChain is essentially the heart and soul for achieving a real RAG system. LangChain is tightly integrated into both the retrieval and generation components of RAG, making the end-to-end workflow possible.

The workflow begins with document retrieval and ingestion. First, you ingest your data into the system. This could be enterprise knowledge bases, sets of PDFs, or other document formats. LangChain facilitates this entire process, acting as the framework that enables retrieval from the enterprise knowledge base. Without LangChain, orchestrating this process would be highly complex.

Once the documents are retrieved, the next step is processing and embeddings. Each sentence, paragraph, or word in the documents is converted into vector embeddings, which are arrays of numbers representing the features of the text. LangChain plays a key role in ensuring that this embedding process is integrated seamlessly into the workflow.

After creating embeddings, these vectors need to be stored for efficient querying. This is where a vector database comes in. LangChain helps manage the storage of document embeddings in the vector database, which is then used to perform similarity searches. When a user submits a query, the query itself is converted into embeddings, which are then compared with the document embeddings to find the most relevant information.

Once the similarity search identifies the relevant documents, the next step is query enhancement and context preparation. LangChain helps combine the original user query with the retrieved information to create an enriched context. This enriched context becomes the prompt that is sent to the language model (LM). By doing this, the LM has all the necessary context to generate accurate and informed responses.

Finally, in the generation phase, the LM creates the response based on the prepared prompt. LangChain manages the interaction with the LM, ensuring that the input is correctly formatted and that the generation process is executed effectively. Essentially, LangChain provides an end-to-end framework that integrates every component of the RAG system—from the initial user query, through retrieval, embedding, similarity search, and context preparation, to the final response generation by the LM.

With this explanation, it becomes clear that LangChain is critical to a RAG system, serving as the backbone that enables seamless integration and smooth operation across all stages of the workflow.

# **J) Prompt Engineering & Fine Tuning**

After getting a good understanding of what a Large Language Model (LM) is, it’s time to understand two more important concepts: prompt engineering and fine tuning. These are crucial from the perspective of generative AI or LMs, so having a clear understanding of them is very important.

Let’s start with prompt engineering. First of all, remember that a prompt is essentially an input. Prompt engineering is the practice of crafting or creating inputs (prompts) for a generative AI model to elicit the best possible output. Prompts are the inputs to a model, and in generative AI, the principle is simple: if your input is poor, your output will be poor; if your input is good, your output will be good. With prompt engineering, you are essentially experimenting with different ways to ask or phrase a question.

For example, you could ask, “What is an antibiotic?” or “Give me some details about antibiotics,” or “Can you summarize antibiotics?” Another example could be giving a statement like, “Here are some lines about antibiotics, is my understanding correct?” These are different ways of prompting the model, and by experimenting with these different prompts and observing the outputs, the model’s performance and understanding can improve over time.

Prompt engineering is a practice that involves using specific keywords, structured sentences, and including examples to guide the AI. Ultimately, the goal is to maximize the AI’s understanding of the task and generate more accurate, relevant, and creative responses. The prompt should be designed in a way that the AI can understand and process it effectively to produce meaningful human-like text. This also requires understanding context; for example, the word “bank” could refer to a riverbank or a financial bank, and the AI must interpret it correctly based on context.

Prompt engineering can be used by both novices and experts. A novice may use trial and error: they provide a prompt, observe the output, refine the prompt, and try again until they get satisfactory results. Experts, on the other hand, know how to craft prompts in a highly precise way to get accurate and creative outputs, often using sophisticated methods informed by a deep understanding of the model.

The second concept is fine tuning. In simple terms, fine tuning is the process of optimizing the performance of a pre-trained AI model to better suit specific tasks or datasets. This involves training the AI on a particular dataset to refine its responses so that they are more aligned with the characteristics of that data. In this process, both the quality and quantity of data are extremely important. Pre-trained models like large language models are trained on massive datasets from books, websites, Wikipedia, and more, which gives them broad knowledge.

Fine tuning requires more computational resources and technical knowledge than prompt engineering because it involves training with large datasets and making adjustments to the model’s parameters. It enables the creation of custom AI models that can perform better on specialized tasks. Fine tuning is not a one-time activity; it’s a continuous process where models are periodically updated as they are exposed to new data.

One challenge with fine tuning is overfitting. Overfitting occurs when a model performs extremely well on the training data but fails to generalize to new, unseen data. Therefore, careful monitoring and adjustments are needed to ensure that the model performs well not just on the trained dataset but also on real-world inputs.

Prompt engineering and fine tuning often go hand in hand. Fine tuning provides a strong baseline performance for the model, optimizing it for the overall task, while prompt engineering optimizes individual interactions, guiding the model to produce the most accurate and creative outputs for specific prompts. Both techniques are crucial for enhancing the performance of generative AI models in various applications and are used together to ensure the model is both well-trained and adaptable to diverse user inputs.

In summary, prompt engineering is about crafting effective inputs to get the best output, while fine tuning is about optimizing the model itself for specific tasks. Both require attention to context, data quality, and experimentation, and when used together, they significantly enhance the capabilities and performance of generative AI models.

# **V) AI Infrastructure**

# **A) What is a GPU ?**

In this video, we will discuss the $1 million question. Why the $1 million question? Just look at how the world is going crazy about AI. The backbone of AI, in terms of infrastructure, is the GPU. Just look at Nvidia shares—they are skyrocketing. This has everyone curious to understand what a GPU is. Most people know about a CPU, which has been used for ages, but why have GPUs suddenly become so important in our lives?

A GPU stands for Graphics Processing Unit. You may wonder why we are talking about a graphics unit in the context of AI. GPUs were originally designed to accelerate graphics rendering, mainly for gaming, which is why some people even call it a Gaming Processing Unit. They were designed to render graphics in applications such as 3D games and animations. The key feature of a GPU is its highly parallel structure. The biggest difference between a CPU and GPU is that CPUs perform computations serially, while GPUs perform computations in parallel. GPUs have a parallel processing architecture, allowing them to process many computations simultaneously. Tasks that take longer on a CPU due to serial processing can be completed much faster on a GPU.

Rendering, also known as image synthesis, is the process of generating a photorealistic or non-photorealistic image from a 2D or 3D model using a computer program. Originally, GPUs were used for this purpose in video games and 3D animation, long before AI became a consideration. However, scientists later realized that the parallel processing power of GPUs could be leveraged in machine learning and AI. Neural networks, for instance, involve multiple interconnected nodes, and GPUs, with their thousands of cores, can handle multiple operations concurrently. This parallelism significantly accelerates the training of complex neural networks.

Moreover, GPUs are more energy-efficient than CPUs for parallel processing tasks. While CPUs can also run multiple cores, their architecture is not as efficient as GPUs for breaking down tasks into multiple subtasks to be executed simultaneously. This results in a better performance-per-watt ratio with GPUs.

There are two main types of GPUs: integrated and dedicated. Integrated GPUs are built on the same chip as the CPU, sharing the same memory and resources. Dedicated GPUs, on the other hand, are separate cards with their own memory and processing power, offering superior performance. GPUs are essential for powering virtual reality and augmented reality applications, where speed and performance are critical for an immersive experience. Wherever there is a large amount of data or graphics to process, GPUs are indispensable.

The technology behind GPUs is rapidly evolving, with constant innovations leading to faster, more efficient, and more powerful GPUs. Major companies in the GPU space include Nvidia, AMD, Intel, and ARM. Currently, Nvidia is the market leader, with AMD catching up. There is significant investment in the GPU industry due to its critical role in AI and graphics-intensive applications.

To summarize, CPUs (Central Processing Units) typically have a few cores—ranging from 8 to 100—and are optimized for low-latency, serial processing tasks. GPUs, in contrast, have hundreds or thousands of cores, optimized for high-throughput, parallel processing. CPUs perform well for tasks requiring sequential computation, while GPUs excel at parallel processing tasks such as graphics rendering, 3D games, machine learning, and deep learning. A CPU processes tasks one at a time, whereas a GPU breaks tasks into smaller subtasks and processes them simultaneously, making it ideal for operations that require high concurrency.

Traditional programming was mostly written for CPUs, so additional software and programming are needed to utilize GPUs efficiently for parallel execution. This is an important consideration for developers who want to leverage GPU power. With this, you now have a clear understanding of what a GPU is. You can also look up images of Nvidia processors to see how GPUs look in the real world. In the next video, a demo will clearly illustrate the difference between CPU and GPU.

# **B) Demo: CPU Vs GPU**

Hello and welcome. I’d like to share a video that was an eye-opener for me. When I was investigating the difference between a CPU and a GPU, I realized that while most of us are familiar with what a CPU is, many people might not be aware of what a GPU does. In the previous lecture, we discussed the theoretical differences between a CPU and a GPU, but now I want to show you a demo that illustrates it visually.

The video I’m sharing is from Nvidia and was uploaded almost ten years ago when they introduced GPUs to the world. In this demo, the presenter paints a picture of how a CPU operates as a series of discrete actions performed sequentially, one after the other. Then, he compares it to how a GPU works with thousands of cores running in parallel.

In the demonstration, when the trigger is hit, 2,100 gallons of air go through accumulators, out through valves, and into 1,100 tubes. Each tube contains a paintball, which flies across seven feet of space and reaches its target in just 80 milliseconds. The aim is to paint the Mona Lisa. While a CPU would handle tasks sequentially, this setup represents a GPU’s parallel processing, where thousands of operations occur simultaneously to achieve a fast, coordinated result.

This demo is a clear and engaging way to visualize the power of GPUs compared to CPUs. It highlights how GPUs, with thousands of cores running in parallel, can perform complex tasks much faster than CPUs, which handle tasks sequentially. Thanks for watching, and I hope you enjoyed the demonstration as much as I did.

# **C) What is RDMA Cluster Network**

After having a good look at High-Performance Computing (HPC), it is time to explore RDMA cluster networks. You might recall that HPC involves using multiple computers in a cluster to perform tasks faster than a single computer could. By adding thousands of computers to a cluster, you can complete more work in less time. In HPC, these computers are interconnected, often with the fastest CPUs and GPUs, but a slow network can become a major bottleneck. To address this, we have RDMA cluster networks.

RDMA stands for Remote Direct Memory Access. Breaking it down: “Remote” means accessing memory on a different computer rather than locally, and “Direct Memory Access” means the data transfer happens directly between memories without involving the CPU or operating system. RDMA cluster networks are specialized network configurations designed specifically for HPC to ensure high-speed communication between cluster nodes.

To understand the benefit of RDMA, consider how two computers normally communicate. Data from one application on a computer passes through several layers—such as the OS, kernel, and network interface card (NIC)—travels over the network, and then passes up the layers of the receiving computer to reach memory. This process introduces latency and slows down performance. RDMA bypasses these layers, allowing direct memory-to-memory data transfer between computers. This reduces CPU overhead, lowers latency, and increases throughput, which is crucial for tasks requiring rapid data processing, such as AI and big data analytics.

RDMA networks are known for their extreme low latency and high throughput, making them ideal for HPC applications. They can be implemented over various network protocols, including InfiniBand, which has been used in Oracle Exadata systems for years. More recently, RDMA over Converged Ethernet (RoCE or Rocky) enables RDMA over Ethernet, and iWARP provides RDMA over wide area networks. These implementations make RDMA highly versatile and adaptable to different environments.

Another significant advantage of RDMA networks is scalability. HPC clusters often grow over time, and RDMA supports seamless expansion, making it suitable for research and enterprise settings where computing demands can increase. RDMA is also beneficial in clustered databases and storage systems, enabling rapid and efficient data transfer between nodes, which is essential for cloud computing, parallel computing, scientific simulations, real-time analytics, and other HPC applications.

RDMA is also energy-efficient. By bypassing the CPU and reducing processing overhead, it optimizes data transfer and consumes less energy than traditional networks. However, there are cost considerations. RDMA cluster networks require specialized hardware and maintenance, making them more expensive than traditional networking solutions. While the performance benefits are significant, implementing RDMA requires investment in infrastructure.

In summary, high-performance computing requires a fast network to prevent bottlenecks, and RDMA cluster networks provide the solution. RDMA allows one computer’s memory to communicate directly with another’s, bypassing OS layers, CPUs, and NICs. This direct memory access improves speed, efficiency, and scalability, making RDMA essential for modern HPC and cloud computing applications.

# **VI) OpenAI / ChatGPT / API**

# **A) What is OpenAI ?**

Hello and welcome. As we are learning about Generative AI, or Gen AI, it’s not possible to continue without talking about OpenAI. I consider OpenAI to be a true game changer in the field of generative AI. Now, you may ask: What exactly is OpenAI? You might have heard terms like OpenAI, ChatGPT, and others. This is what I’m going to demystify for you. OpenAI is fundamentally an AI research lab. It started as a research organization with a key mission: to develop and promote friendly AI that benefits humanity as a whole. Their intention was to create intelligent computers and AI systems that help humans. In the beginning, they were not focused on making money or commercializing their research; they simply wanted to help humanity by advancing AI technology.

OpenAI was founded in 2015 by several well-known names, including Elon Musk, who many of you know as the owner of Tesla and Twitter/X; Sam Altman, Greg Brockman, Ilya Sutskever, and Wojciech Zaremba, among others. Later, in 2022, OpenAI released what would become one of the most revolutionary AI systems ever created: ChatGPT. For now, just understand that OpenAI is the company, and ChatGPT is one of its most famous products. When ChatGPT was released in November 2022, Elon Musk became irritated with Sam Altman and the direction the company was taking. Musk publicly stated that the chatbot had accelerated a dangerous race to develop powerful AI systems. He felt that the direction was shifting toward replacing humans, which he did not support.

Musk also disagreed with the fact that OpenAI originally positioned itself as a nonprofit organization promising open access to AI research, but later created a for-profit entity and raised billions of dollars — especially from Microsoft. He questioned how a nonprofit he co-founded transformed into a commercial model backed by massive funding. Because of all these disagreements, Musk eventually left the organization.

Despite these conflicts, OpenAI has achieved remarkable feats. It has developed some of the most advanced AI models ever created, including the GPT (Generative Pre-trained Transformer) series — GPT-1, GPT-2, GPT-3, GPT-4, and potentially GPT-5 depending on when you are watching this. Another key aspect of OpenAI’s work is its focus on ethics and safety. They put significant effort into ensuring that AI is used ethically and securely, reducing risks associated with highly capable AI systems. They also started with a philosophy of open collaboration — sharing research and findings openly. However, this openness also created risks, because advanced AI models could be misused. This forced OpenAI to introduce controlled release strategies to ensure safety.

As the organization grew, it realized that being purely open and nonprofit was not sustainable. A company needs funding to operate, hire talent, and build large-scale infrastructure. This led to the creation of commercial products and a very smart business model built around AI APIs. OpenAI charges for the input tokens (the prompt you give, such as “write a story”) and the output tokens (the AI’s response). This token-based billing applies to API usage for models like GPT-3 and GPT-4 for tasks such as conversation, summarization, and translation.

One of the biggest turning points for OpenAI was its partnership with Microsoft, which invested $13 billion into the company. Today, in Microsoft Azure, you can find Azure OpenAI Service, which allows enterprises to use OpenAI models in a secure, scalable way. This partnership has been extremely powerful and will likely continue for years. OpenAI also has other partnerships, but Microsoft remains its biggest and most significant collaborator.

With this, you now have a strong end-to-end understanding of what OpenAI is, how it started, the challenges it faced, how its mission evolved over time, and how it became one of the most influential forces in modern AI.

# **B) What is ChatGPT ?**

So I believe now you have a good understanding of what OpenAI is as a company. You understood its origins, its mission, and its evolution. Then we looked at GPT, which forms the backbone of ChatGPT. We explored what GPT means—Generative Pre-trained Transformer—and we also took a deep dive into how it works. Now the natural question is: What is ChatGPT? As the name suggests, “ChatGPT” can be broken down into two parts: Chat and GPT. GPT, as you already know, refers to the underlying generative transformer technology, and “Chat” indicates its interactive nature. But it’s not just a typical chatbot; it represents a highly advanced form of conversational AI.

ChatGPT is built on top of GPT technology. In the previous explanation, you learned extensively about GPT, and now you can connect that understanding here. GPT is an AI model designed specifically to generate text, not to predict numerical outputs like predictive AI systems. It belongs to the field of generative AI because it creates content on the fly—whether you're asking it to write a story, a poem, or a piece of code. It produces new text dynamically based on your prompt.

This capability comes from the fact that ChatGPT has been trained on a large and diverse corpus of internet text. When the model was being created, OpenAI fed it a huge amount of data—including news articles, blogs, books, encyclopedia content like Wikipedia, and countless other sources. Because of this broad training data, the model is able not only to generate text but also to understand the context of what is being asked. This is one of the biggest differences between generative models and traditional predictive models. For example, it can differentiate between “money in the bank” and “bank of the river” based purely on context. It can answer questions, write essays, compose poetry, or even generate complete code.

ChatGPT functions through a conversational interface designed to simulate human-like dialogue. To use it, you simply visit chat.openai.com, choose the model (for example GPT-3.5 or GPT-4), and type your question. It responds immediately with a context-aware answer. The interface is entirely based on natural language processing, which allows the model to interpret your input as a human would.

Now, ChatGPT does have learning capabilities, but not in the way many people assume. It does not learn from individual user interactions. It won’t store personal conversations or improve itself based on your specific inputs. Only OpenAI engineers can update or retrain the model. So even if someone asks irrelevant or harmful questions, ChatGPT will not adopt or learn from those interactions. Its improvement comes strictly from controlled training by OpenAI.

ChatGPT can be used in many real-world applications. Since it's an advanced conversational system, it can power customer service chatbots, automated support assistants, tutoring systems for educational topics, or even specialized domain-focused assistants trained on specific content. It can also support content creation, whether that means generating text, summarizing information, or creating images when connected with image-generation models.

OpenAI has implemented strong safeguards and guardrails to prevent ChatGPT from generating harmful or inappropriate content. They understand that even children might use ChatGPT, so if someone attempts to ask for adult, abusive, or unsafe content, the model will refuse and redirect the user. Safety and ethics are a core part of how ChatGPT was designed.

A major advantage for developers is that ChatGPT or GPT models can be integrated into custom applications using APIs. For example, you can embed GPT capabilities into CRM systems, ERP tools, mobile apps, or enterprise software to enhance them with intelligent, conversational features. This allows organizations to tailor ChatGPT-based functionality to their specific business needs.

ChatGPT continues to evolve with ongoing research and development. New models such as GPT-4 have already appeared, and GPT-5 is under development and expected to be released soon. OpenAI’s partnership with Microsoft—which involved a massive $13 billion investment—means both companies are jointly advancing this technology. As research continues, more powerful, safer, and more capable models will be released in the coming years.

The goal here was to give you a solid, complete understanding of what ChatGPT is and how it works.

# **C) Demo on chatgpt**

Now the time has come to actually put generative AI—specifically ChatGPT—to the test. The process is very simple. You begin by logging in to chat.openai.com. Once you reach the login page, you’ll see two main options: you can either sign up if you are a new user or log in if you already have an account. To sign up, you simply provide a username, password, and email ID. Alternatively, you can make things easier by signing in using your Google, Microsoft, or Apple credentials. For this demo, to keep everything straightforward, I’ll log in using my Google account. If you’re doing this for the first time, Google will prompt you for your Gmail credentials, and once authenticated, you’ll be taken into the ChatGPT interface.

Inside ChatGPT, most users will initially see GPT-3.5, which is the free version. It handles everyday tasks quite well. Alongside it, there is GPT-4, which includes additional features like browsing and more advanced analysis, though it comes with usage limits and is part of the ChatGPT Plus subscription. For this demonstration, I will choose GPT-3.5 and begin with a simple question. I ask: “Who is the Prime Minister of India? Give me the bullet points on his accomplishments.” Immediately, ChatGPT begins generating the response. Notice how fast it works—it is not browsing the internet but generating the answer based on its trained knowledge. Because GPT-3.5 was trained up to 2022, it responds by saying: “As of my last update in January 2022…” and then provides the answer. It identifies Narendra Modi as the Prime Minister and lists accomplishments such as Digital India, Jan Dhan Yojana, foreign policy initiatives, and more. The answer is quick and well-structured.

Next, I try something different. I request: “Write me a 50-word story on the topic ‘My Best Friend.’” And instantly, ChatGPT creates a perfectly formed 50-word story. To push it further, I ask: “Write me a poem for my wife, whom I met 20 years ago and who loves cooking.” Here, the prompt provides context—how long ago I met her and what she enjoys. The model then takes these cues and generates a beautiful, personalized poem. That’s the power of generative AI: it doesn’t just answer; it creates.

I can go even further and ask it to generate code. For example, I type: “Write me a Python code to print the Fibonacci series.” The Fibonacci sequence is a piece of logic that may take time to write manually if you're not familiar with it. Yet ChatGPT produces the complete Python code within seconds. It can generate stories, poems, essays, structured information, and even functional code—all on demand.

This is where we emphasize again: ChatGPT is not a predictive system in the traditional sense, and it is not like Google. Google searches the internet for relevant documents. ChatGPT does not. It consults its trained internal model and generates new content dynamically based on patterns learned during training. That is why it belongs to the category of generative AI. It produces fresh content, in real time, tailored to the prompt you give.

So, with this demonstration, you now have a clear understanding of how ChatGPT works in real-time scenarios—from answering factual questions to creative writing, from generating poetry to writing code. Thanks for watching.

# **D) Time to reach 100M users**

This is going to be a very quick video. All I want to highlight here is why I call ChatGPT and OpenAI true game changers. And the simplest way to understand this is by looking at the sheer popularity of ChatGPT. On this slide, you can see a comparison of how long it took various major companies to reach 100 million users, and the numbers clearly speak for themselves.

Let’s start with Google Translate. It took Google Translate around 78 months to reach 100 million users. Then we have Uber—something almost everyone uses for taxi rides—and Uber took about 70 months to hit the same milestone. Moving on, there is Telegram, a messaging app similar to WhatsApp, and Telegram reached 100 million users in around 60 months.

Next is Spotify, which many people use for online music and streaming. Spotify took approximately 55 months. After that, we look at Pinterest, which reached the 100-million mark in about 40 months. Instagram, another extremely popular platform from Meta, achieved this milestone in roughly 30 months.

Then we come to TikTok. TikTok, which you can think of as similar to YouTube Shorts in concept, grew incredibly fast and reached 100 million users in around nine months. And now comes the most surprising part—the reason I call ChatGPT a breakthrough unlike anything before.

As soon as ChatGPT was released, it took only two months—just two—to reach 100 million users. This is an unprecedented rate of adoption. And the growth hasn’t slowed down; it is continuing at a rapid pace, setting new records in the world of technology.

So, I believe this gives you a clear understanding of why I keep saying that ChatGPT and OpenAI are absolute game changers in the field of generative AI.

# **E) Get an understanding of Various OpenAI Models**

So if you ask me today what the heart and soul of OpenAI is, I would say it very confidently:
the heart and soul of OpenAI is its models — the kind of powerful models they have introduced to the world.

All these models are built with a specific use case in mind, and each one is designed with different capabilities. OpenAI clearly states that their ecosystem is powered by a diverse set of models, each having its own strengths and its own price point.
That means every model has a particular purpose, a particular capability, and also a particular way in which you are billed when you use it.

Another major advantage is that you’re not restricted to only pre-trained models.
OpenAI also gives you the ability to customize models, and the process used for customization is what we call fine-tuning.
We already discussed what fine-tuning means in one of the earlier videos, but the important point here is: if you want a model to work for a very specific use case, fine-tuning is the way forward.

Now, let’s talk about some of the most popular models that OpenAI provides today.
The most recent flagship model is GPT-4 Turbo.
They first launched GPT-4, and later, they released GPT-4 Turbo as an advanced, improved version.

If we click on GPT-4 Turbo, you will notice that it's a large multimodal model.
“Multimodal” means it can accept text as well as image inputs, and it always gives text as output.
It is designed to solve difficult problems with higher accuracy.

Another important thing you should always check is the training data cut-off.
The more recent the training date, the more updated the model is.
For example, GPT-4 Turbo has been trained till December 2023, which is extremely recent.
If you scroll below, GPT-4 (the earlier version) was trained till September 2021.

You will also observe that the number of tokens supported keeps increasing as new models come out.
And don’t worry — we will discuss what tokens are in a separate video.

Next, OpenAI introduced GPT-3.5, which, as we’ve seen in earlier videos, is the free model available for everyone.

Then comes DALL·E.
DALL·E is a model that can generate and edit images based on natural language prompts.
It is incredibly powerful — you can simply describe an image in detail, for example:
“A black dog with black eyes sitting beside a white cat with blue eyes on a beach,”
and DALL·E will generate a picture exactly matching your description.

Following that, we have TTS, which stands for Text-to-Speech.
This model converts written text into natural-sounding audio.
OpenAI provides different voice templates, and you can choose the one you like to generate speech from text.

On the other hand, Whisper works in the opposite direction.
Whisper is a model that converts audio into text.
So TTS is text → audio, and Whisper is audio → text.

Next is the concept of embeddings, something we already discussed.
Embedding models convert text into numerical representations, and those numbers capture the meaning or features of the text.
These embeddings are widely used in search, recommendations, clustering, and RAG systems.

Then we come to moderation models.
OpenAI takes ethics and safety very seriously.
Moderation models ensure that the content you provide does not violate the policies.
They can detect whether your prompt contains hate, violence, sexual content, self-harm topics, or anything that is restricted.
This ensures the platform remains safe for everyone.

After that, we have the older GPT Base models.
These were trained to understand and generate natural language or even code, but they were not trained to follow instructions the way newer models are.
Most people won’t be using these anymore, but they are still part of OpenAI’s history.

Finally, there is the Deprecated section.
Whenever a model is retired or shut down, OpenAI moves it into this category.
You can scroll through the list to see which models have been discontinued along with their shutdown dates.

So the intention of this video is simple:
OpenAI is much more than just ChatGPT.
It is an ecosystem of many specialized models — image models, speech models, text models, embeddings, moderation, and more.
Knowing these gives you a complete picture of what OpenAI truly offers.

With this, you now have a solid understanding of the different models available in OpenAI.

# **F) GPT-3 vs GPT-4**

Hi folks, welcome back. In this video, we’re going to take a detailed look at the differences between GPT-3 and GPT-4. Many people often ask me what the real difference is between these two versions, because OpenAI came up with GPT and then released multiple versions of it, which we discussed in the previous lecture. So first of all, GPT-3 and GPT-4 are both iterations of OpenAI’s Generative Pre-trained Transformer—GPT—which we know is the architecture designed by OpenAI. Even though both are versions of the same technology, there are several key differences. Some of those differences are visually represented in this figure, and thanks to V9 Digital for that illustration. You can see that GPT-3 accepts only text prompts, meaning you can only provide text as input, whereas GPT-4 accepts both text and image prompts, which is definitely an advancement. GPT-3 was already considered creative, but GPT-4 is even more creative, and you can clearly feel the difference when you ask it to write a poem, a story, or any creative content. GPT-3 tends to hallucinate a lot of facts and opinions—it generalizes quite heavily—while GPT-4 still hallucinates but much less. Another example is the bar exam. GPT-3 barely passed the bar exam, whereas GPT-4 actually aced it, performing extremely well. GPT-3 also needed a lot of steering and prompting from developers—you had to guide it a lot to get good results—but GPT-4 is more steerable by design and handles conversations and developer prompts more effectively.

One of the biggest differences lies in model size and complexity. GPT-3 has around 175 billion parameters, making it one of the largest language models of its time. GPT-4 is even larger, with significantly more parameters, and the trend will continue with future versions like GPT-5 and GPT-6, where parameter counts will increase even further. Then comes performance and accuracy. GPT-3 sometimes struggles with complex reasoning tasks and can produce less accurate results because it was relatively early in its development. GPT-4, on the other hand, has matured and shows much better performance in understanding context. Remember, in transformer architecture, context is key—this is related to the concept of self-attention. GPT-4 delivers more accurate and contextually relevant outputs. When it comes to training data and knowledge, GPT-3 was trained on a vast corpus of text up to its cutoff date in 2020, using sources like Wikipedia and The Telegraph. GPT-4 was trained on many more sources, much larger datasets, and far more diverse information, which expands its knowledge range and improves contextual understanding. As we always discuss, context matters—for example, distinguishing between “bank” as in money vs. “bank” as in river. GPT-3 has a good understanding of context but tends to lose coherence during longer conversations or more complicated queries, whereas GPT-4 maintains context for longer and handles complex queries more effectively.

Another major difference is modality. GPT-3 is purely text-based—you provide text, and it outputs text. GPT-4 is multimodal, meaning it can understand images in addition to text. In terms of applications, GPT-3 has been widely used across industries for content creation, letter writing, poem writing, storytelling, coding, customer service, corrections, and many other tasks. GPT-4 builds on all these applications but provides improved performance, making it more effective and versatile, especially for complex tasks. When we compare error rates and reliability, GPT-3—though advanced for its time—still had higher error rates for complex tasks. GPT-4 reduces those error rates significantly, which directly increases reliability. Lower error rates always lead to more dependable results. So overall, these are the key differences between GPT-3 and GPT-4. Thanks for watching.

# **G) New: What is GPT-4o ?**

As always, my aim is to keep you updated with all the modifications, improvements, and innovations happening in the world of OpenAI and Azure OpenAI. In line with that, I’ve prepared a new video focusing on two recently released models from OpenAI — GPT-4O and GPT-4 Mini. These models have created a lot of excitement, and it’s important to understand what they stand for and what makes them special.

Let’s begin with GPT-4O.

GPT-4O is essentially an advanced, optimized, and more powerful version of GPT-4. If you recall the evolution — we started from GPT-3.5, then GPT-3.5 Turbo, then the breakthrough GPT-4, and now we have GPT-4O. The immediate question that everyone asks is: What does the “O” stand for?

The “O” stands for Omni.

And what does Omni mean? Omni simply refers to “everywhere” or “everything.” For example, when we say “God is omnipresent,” we mean God is everywhere. That’s exactly the inspiration behind the Omni model — the goal was to enable the model to handle “everything” that previous versions like GPT-3 or GPT-4 could not handle.

Released in May 2024, GPT-4O is quite recent. Its biggest strength lies in its multimodal capability. When we say “multimodal input,” it means the model can accept any combination of:

Text

Audio

Images

Video

That’s the power of GPT-4O. You can give it input in any of these modes, and it can generate output as text, audio, or images. The only limitation as of now is that it does not generate video output, but both video and audio inputs are fully supported.

Another major improvement in GPT-4O is its enhanced context length. Context, as you know, refers to the model’s ability to remember your previous prompts, understand the flow of the conversation, and interpret ambiguous words based on usage (like “bank” meaning money vs. river). GPT-4O can maintain context much longer and much more accurately than GPT-3 or GPT-3.5.

A very interesting fact, which many people don’t know, is that GPT-4O is not larger than GPT-4 in terms of parameters. GPT-4 allegedly had around 175 billion parameters, while GPT-4O has around 12 billion parameters. Yet, GPT-4O performs better. How? Because “O” also stands for Optimized. It’s a heavily fine-tuned and highly efficient model that delivers better performance with fewer parameters. That’s the beauty of optimization.

This optimization reflects directly in accuracy. GPT-4O provides more accurate and more relevant responses, significantly reducing incorrect or confusing outputs that older models sometimes produced. This is supported by benchmark tests.

OpenAI compared GPT-4O with other major models — GPT-4 Turbo, earlier GPT-4 versions, Claude, Gemini, and more — across different benchmarks like:

MLU

GPQA

Math reasoning

Human evaluation

MGSM

DOP (which checks performance using F1 scores — precision + recall)

In almost every category, GPT-4O comes out on top, followed by GPT-4 Turbo. This clearly shows how optimized and powerful GPT-4O is compared to predecessors and competitors.

Another major highlight of GPT-4O is its lightning-fast speed. The first time I tested it, I was genuinely surprised — the speed was unbelievable. Compared to GPT-4, GPT-4O responds almost instantly. That’s why it’s highly suitable for real-time applications where latency matters.

What shocked me even more was the human-like response time. According to OpenAI, GPT-4O can respond to audio in as little as 232 milliseconds, with an average of around 320 milliseconds — almost the same as how two humans respond to each other in a natural conversation. That’s a massive achievement in AI.

On top of all this performance improvement, GPT-4O is also 50% cheaper than GPT-4 models.

For example:

GPT-4 Turbo cost $10 per 1M input tokens → GPT-4O costs $5 per 1M input tokens.

GPT-4 Turbo output cost $30 per 1M tokens → GPT-4O costs $15 per 1M.

So not only do you get a faster and more accurate model, but you also get it at half the cost.

# **H) New: Demo: GPT-4 Vs GPT-4o**

So when I was comparing the ChatGPT-4 model and the ChatGPT-4O model, I had mentioned to you guys that the 4O model is much, much faster. I think we even said that it is almost twice as fast and also 50% cheaper. So I thought, why don’t I actually put this to the test? What if I execute the same input or the same prompt in ChatGPT-4 and then run that exact same prompt in ChatGPT-4O? What kind of response difference would I actually see?

So that’s exactly what I did. If you look here, on the left-hand side, I have ChatGPT-4 opened up, and on the right-hand side, I have ChatGPT-4O. I am giving both of them the exact same question or the exact same prompt: “Write a one-page story on a guy who loved to create content on Udemy for IT courses.” The same prompt is written on the left, and the same prompt is written on the right.

Now for fun, I decided to execute the prompt first on GPT-4 and immediately after that on GPT-4O. So I run it first on GPT-4, and quickly I trigger the same on GPT-4O as well. And here’s what happens. GPT-4O starts generating the answer almost instantly. It begins typing, continues smoothly, and within seconds, it is completely done creating the one-page story. Meanwhile, ChatGPT-4 is still typing. It’s still generating, still going, still processing, still completing the story, while GPT-4O has already finished the entire response.

And that’s the difference. Even though this is a very simple, very layman-level example, it clearly shows that what OpenAI says about the GPT-4O model being significantly faster is absolutely true. It really is much faster than GPT-4, and this small demonstration makes it very easy to understand the difference in real time.

# **I) New: What is GPT-4o Mini?**

So after taking a look at the GPT-4 model, it’s time to take a look into the GPT-4 Mini model. Now, the thing is, I was actually wondering—when OpenAI brought the GPT-4 model in May 2024, what was the urgent need to bring GPT-4 Mini so soon? And the reason, I’ll tell you, is competition. We all know that OpenAI is doing extremely well in the field of LLMs, the large language models, but what is happening in the market is that a lot of players are now coming with SLMs. You might not have heard the term much, but you can think of it as a Small Language Model. For many use cases, you actually don’t need a large language model. You don’t need that level of computational resources, and you don’t need that level of GPUs. Think about simple use cases for small to mid-sized companies— they don’t require such heavy computation. That is where players like Gemini Flash, and companies like Anthropic, come into the picture. Anthropic came up with the Claude Haiku model, and this Claude Haiku model was giving good competition to GPT-3.5 Turbo because that was also a smaller and cheaper model. The bigger GPT-4 models are still quite expensive, especially for small to mid-sized companies.

So that is where OpenAI introduced GPT-4 Mini—to compete with these smaller models. If you notice, the comparison here is not between GPT-4 Mini and the big GPT-4 models. Instead, the comparison is between the smaller-sized models: GPT-3.5 Turbo, Gemini Flash, and GPT-4 Mini. Yes, there is a slight comparison with GPT-4 as well, but the main focus is on the small models. GPT-4 Mini stands for “small”—it is a compact, smaller, and efficient version of the larger GPT-4 and GPT-4 Turbo models. It is specifically designed for deployment in resource-constrained environments, meaning places where you don’t have enough computational power or where you don’t even need much computational power. In those situations, GPT-4 Mini becomes very useful. The release date of GPT-4 Mini is quite recent—July 2024. As I said, it's compact, and it's designed to run efficiently on hardware with limited computational resources.

It also provides faster response times compared to the full-sized GPT models, making it ideal for time-sensitive tasks. Now you may ask, why does it give faster responses? The reason is simple—it has fewer parameters and much less computation to do. When you query the model, there is less internal processing required, so the response naturally comes faster. And when we look at the model evaluation scores where we compare GPT-4 Mini, Gemini Flash, and other models across different benchmarks that we discussed in the previous video, you will see that GPT-4 Mini actually comes first in most of the cases, and Gemini Flash comes very close in many of those benchmarks.

Another important aspect is power efficiency. GPT-4 Mini prioritizes lower power consumption, unlike the large models that require significant energy. Everywhere today, there is discussion about being energy-efficient or creating greener systems. Large language models are powerful, but they aren’t energy-efficient because they require massive computational resources and high power consumption. Smaller language models, on the other hand, need much fewer GPUs and much less power, making them a good fit for companies aiming to reduce operational costs and carbon footprint.

A very important point is that GPT-4 Mini can be installed on local devices. Even if you don’t have internet access or cloud infrastructure, these models can run locally. They are designed to function independently of server-based computation. Larger models rely heavily on cloud-based systems—the query has to travel to the server, the model processes it, and the response comes back. But GPT-4 Mini can be deployed on devices like smartphones, tablets, IoT devices, and even wearable devices. This is where we talk about edge deployment, and GPT-4 Mini fits that requirement perfectly.

It is also extremely cost-effective. The operational costs are lower because the resource requirements are reduced. If OpenAI is not spending as much on massive GPU clusters for this model, they don’t need to charge customers high prices. For example, when we compare the price of a large model like GPT-4 Turbo with GPT-4 Mini, the difference is huge. If we were paying around $5 for 1 million tokens with GPT-4 Turbo, with GPT-4 Mini we are paying around $0.15 per million tokens. For a small to mid-sized company, this is a massive difference. Why would they pay so much more when their use case doesn’t require a big model? That is exactly why OpenAI realized the need to bring a smaller model to the market, one that can compete with all these new compact models.

And OpenAI clearly says that you can easily replace GPT-3.5 Turbo with GPT-4 Mini. So it's really simple—wherever in your code you’re using model = GPT-3.5-turbo, you can simply replace it with GPT-4 Mini, and you’ll get faster responses and pay less money.

Yes, there is also a difference in training. The heart and soul of any AI or generative AI system is the training process—the data used and how the model is trained. GPT-4 Mini is easier and faster to train compared to large-scale GPT models that need extensive datasets and powerful GPUs. Since the model is smaller, the training is naturally more simplified.

What I personally feel is that the market will eventually split into two segments: SLMs and LLMs. LLMs will include the big models like GPT-4, GPT-4 Turbo, and others. SLMs will include GPT-4 Mini and similar models from other companies. These two streams will run parallel, and a lot of competition will come from Google’s Gemini series as well. So keep watching this space.

# **J) What are tokens ?**

Okay, now it's time to talk about tokens. So guys, remember that tokens are kind of the backbone for any large language model. When we discuss OpenAI models, they process text using tokens. Whatever text you provide — it could be a sentence, a group of words, or even individual letters — is always broken down into tokens because the LLMs understand and work with tokens. Based on these tokens, the models can understand the context of what you want to convey and generate responses. Essentially, the models learn the statistical relationships between tokens, so they are always analyzing and generating text in terms of tokens. That’s why tokens are such an important concept and need to be understood well. A token is simply the basic unit of text — it can be a word, part of a word, or even punctuation. Tokens are the building blocks that models like ChatGPT analyze and generate. For example, if you give the input “hello how are you,” the text is broken into tokens: “hello” is one token, “how are” could be another token, and “you” is a separate token. Spaces are also considered, so this input ends up being about four tokens. The process of breaking text into tokens is called tokenization. 

Large language models work on these tokens rather than the original words, so when you input a sentence like “Hey GPT, how’s the weather today?” the model tokenizes it, processes the tokens to understand the context, and then generates output tokens. These output tokens are then converted back into human-readable text because if the model gave only tokens, humans wouldn’t understand it. Models have a fixed vocabulary size, meaning they can recognize only a certain number of unique tokens. Using tokens allows LLMs to efficiently process large amounts of text by breaking complex structures into manageable pieces. Understanding tokens is important because, in the next lecture, we’ll talk about pricing, and you’ll see that tokens are the key behind it — you pay for the input tokens you send and the output tokens the model generates. This is the foundation of the pricing model that OpenAI has developed. That’s why understanding tokens is really, really important.

# **K) Pricing Model for ChatGPT**

Hey, so it's time to talk about the pricing for OpenAI. The pricing information is available at openai.com/pricing, and it is designed to be simple, transparent, and flexible. Essentially, you only pay for what you use, making it a pay-as-you-go model. This is where the concept of tokens, which we discussed in the previous lecture, becomes very important, because the entire pricing model revolves around tokens. Each OpenAI language model has different capabilities and corresponding price points, and prices are typically measured in units of 1,000 or 1 million tokens. Tokens can be thought of as pieces of words, where 1,000 tokens roughly correspond to 750 words. For example, a paragraph of about 35 words would be around 35 tokens. It is crucial to understand tokens because you pay both for the input you provide to the model and for the output generated by the model. For instance, if you are using the GPT-4 Turbo model, you are charged $10 per 1 million input tokens and $30 per 1 million output tokens. Different models have different pricing; GPT-4 standard may charge $30 for input and $60 for output, whereas GPT-3.5 is cheaper at about $5 per 1 million tokens. In addition to language models, OpenAI also offers fine-tuning models, embedding models, and image models, each with their own pricing structure. 

Another important point is that API access is billed separately from ChatGPT subscriptions. While ChatGPT 3.5 is free to use via chat.openai.com, usage of GPT-4 or other advanced models requires payment, and API calls also have separate charges. ChatGPT Plus subscriptions, which cost $20 per month, cover usage via chat.openai.com but do not include API access, which must be paid for separately. This separation of API access and subscription plans highlights why understanding tokens is critical for cost management. OpenAI also provides a billing dashboard where users can top up funds, set budgets, and track usage. Overall, by understanding tokens and the separate pricing for API calls versus subscription use, users can manage their costs effectively and select the appropriate model for their needs. With this, we conclude the lecture on OpenAI pricing. Thanks for watching.

# **L) Demo: How to make API Calls with OPENAI APIs**

Hello and welcome. I'd say this is the most important lecture of the series of lectures that we are having. Why? Because it talks about the prerequisites that are needed to make the API calls for OpenAI. Now, when I was actually studying, it took me a while to understand how the whole process works. The best thing I would suggest is that it is actually pretty simple, even though it’s not well documented anywhere.

You just need to go to platform.openai.com. Once you log in with your credentials, the next step is to go to Settings, and then to Billing. As I mentioned earlier, making API calls is a paid feature, so you need to top up your balance. Initially, your balance will be zero dollars. For practice purposes, you can top it up with a small amount, like $10, which is what I did.

The next important step is working with API keys. In all the demos we’ll do, we will make API calls to different models, and for authentication, you first need to generate an API key. OpenAI has introduced a concept called Project API keys. Previously, there were normal user API keys, but they are deprecating that model. Now, you can click on Create New Key, give it an optional name like "Testing APIs," select the default project, and click Create a Secret Key.

As part of the process, OpenAI will verify that you are human. They may ask you to complete a puzzle, such as rotating an object in the direction a hand is pointing. This verification usually has to be completed multiple times. Once verified, you will be given a secret key. It is very important to copy and save this key safely, because you won’t be able to retrieve it again.

Next, open your environment for coding. For example, in Anaconda, go to a Jupyter notebook and create a small text file named openai.env. Inside this file, add an entry with your secret key. Make sure to assign it to the variable OPENAI_API_KEY because we will be referencing this variable in the upcoming demos to make API calls to OpenAI. This ensures that your API key is securely stored and accessible in your code.

So, to summarize, the main prerequisites for making API calls using OpenAI are: first, top up your balance by going into Settings → Billing; second, create your secret key, keep it safe, and store it in an openai.env file with the variable OPENAI_API_KEY.

With these steps completed, you are fully prepared to make API calls to OpenAI in the next demos.

# **M) Demo: Make a simple API Call**

See how you can work and make simple API calls to OpenAI. Before you start, make sure that the OpenAI.env file is already created, which contains your secret key, and that you have topped up your credit under the billing section. If your balance shows $0, just top it up by something like $10 for practice purposes.

The first step is to install the required modules for making API calls. The OpenAI Python module might not be installed by default, so in your Jupyter notebook, open a new kernel and run pip install openai. If it’s already installed, it will indicate “requirement already satisfied.”

Next, we will make a simple API call. For example, you can ask: “Who is the Prime Minister of India?” and request some bullet points on his achievements, similar to how you would interact with ChatGPT. To do this in Python, you need a few libraries: openai, load_dotenv, and os. The environment file we created, OpenAI.env, will be loaded using load_dotenv(). Then, the secret key stored as OPENAI_API_KEY in the environment file is accessed using os.environ.get("OPENAI_API_KEY"). This key is passed to the OpenAI client for authentication.

Once authenticated, you can create a response using the client. For example:

client = openai
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Who is the Prime Minister of India? Give bullet points on his achievements."}]
)
print(response)


Here, you can choose the model you want, like GPT-4, GPT-3.5 Turbo, or GPT-4 Turbo. The API call checks your credentials, authenticates your request, and returns a response. For example, using GPT-4, it may respond: “Prime Minister of India as of my knowledge cutoff October 2021 is Narendra Modi,” followed by notable achievements such as the implementation of GST, the Clean India initiative, and the Jan-Dhan Yojana.

The response also provides token usage: completion tokens, prompt tokens, and total tokens. In this example, the completion tokens were 332, prompt tokens 21, and total tokens 353. Remember, you pay for both input tokens (your prompt) and output tokens (the response).

This demonstrates how easily you can make a simple API call to OpenAI and retrieve information.

# **N) Demo: How to create Embeddings ?**

This would be a video on a demo on embeddings. You might remember that we previously understood what an embedding is. Embedding is a way of converting the features of an object, like a word, into a vector of real numbers. Essentially, you take a word, a document, or even an image, and convert it into several real numbers. For example, if you take the word “Apple,” it could refer to the fruit or the company that makes phones and MacBooks. A machine or an embedding model can analyze the context and produce vectors that capture these meanings.

In this exercise, we’ll test what we’ve learned. For instance, if you convert the words “cat” and “kitten,” their vectors will be closely spaced in the vector space because they are related. A “dog” vector will be farther away because, while it is also an animal, it differs from “cat” and “kitten.” This demonstrates the relationship and distinction between objects in vector space.

To implement this, we call the OpenAI API using client.embeddings.create(). For example, we can input a word like “cat” and use the model text-embedding-3-small. Running this code in Jupyter Notebook generates a large set of real numbers, which are the embeddings for the input word. Each number represents a feature of the word in the context of the model, such as “domesticated,” “animal,” etc.

Since these embeddings often have very high dimensions, comparing them directly can be challenging. To make them easier to analyze, we reduce their dimensions, for example, down to seven dimensions. We can normalize the arrays and print them, which gives a more manageable representation while still preserving the relationships between words.

For example, when we generate embeddings for “cat,” “kitten,” and “dog,” we see that “cat” and “kitten” produce similar vectors, indicating their close relationship, while “dog” shows a different set of values, reflecting its distinction. In a vector space, “cat” and “kitten” would be close, while “dog” would be farther away.

This dimensional reduction allows us to visualize and analyze relationships between words more clearly. The embeddings convert the features of an object into real numbers, which can then be stored in a vector database for further applications like similarity searches, recommendations, or machine learning tasks.

This demo helps illustrate how embedding models work, converting complex features of objects into numerical vectors that capture relationships and context. You should now have a clear understanding of embeddings and their practical application. 

# **O) Demo: Image generation using DALL·E in the API**

This is a quick demo to showcase image generation using DALL·E, and it’s an amazing experience. The process is quite simple. All you’re doing is making a call to OpenAI for image generation, while the rest of the setup remains similar to the previous examples. You first import the necessary libraries, such as openai and load_dot_env. The openai.env file is required to read your API secret key. You access the environment variable using API_KEY and initialize the client with OpenAI.

Next, you create a response by calling client.images.generate(). Unlike text generation, here the model used is dalle-3, which is specifically designed for image generation. The most important part is the prompt, where you describe the image you want to generate. For example, you could request an image of a black furry dog with black eyes and a dog collar, alongside a white cat with blue eyes and a necklace. You also specify the type of image and its size, such as 1024 by 1024 pixels.

Once the request is made, the API generates the image and returns a URL where the image can be viewed. Execution may take a few seconds or minutes, depending on the request, so patience is needed. When you open the URL, you will see the generated image. One interesting observation is that each API call can produce a different version of the image; it is not always identical, which adds creativity and variety.

This demo clearly demonstrates the power of OpenAI, showing how beautifully and easily images can be generated with a simple API call. It highlights the flexibility of DALL·E in converting descriptive prompts into visual content.

# **P) Demo: Convert Speech to Text**

This is a very quick demo on speech-to-text and how you can use OpenAI to convert audio or video files into text. The process is quite straightforward. You can work with MP3 or MP4 files and extract the spoken content into text format. First, you import the necessary OpenAI libraries and access your API key from the environment variable, as we have done in previous demos. Next, you specify the path to your audio or video file. In this demo, an MP4 video of Tom Cruise is used as an example.

You then create a variable called transcription and call client.audio.transcriptions.create(). For speech-to-text, OpenAI provides the Whisper model, which is used here. You pass the model name as whisper and the file path to the audio file. Finally, you print the output using print(transcription.text). The transcription variable contains the text generated from the audio, which is displayed in a readable format.

When you run the code, OpenAI processes the audio file and generates highly accurate text. It not only captures the content but also maintains proper grammar, punctuation, and formatting. Full stops, commas, and question marks are placed correctly, reflecting the spoken language faithfully.

In the demo, the transcription reads: “Tom, your company became prominent on the internet with the release of a fake Tom Cruise video, deep Tom Cruise, that I think attracted like a billion views on TikTok and Instagram. Which leads me to my first question, which is, please can we at TED have our own Tom Cruise video, please?” The output demonstrates the model’s precision and reliability in converting speech to text, making it a powerful tool for audio and video transcription. Thanks for watching.

# **Q) New: What is OpenAI O1 Model ?**

As I promised, whenever there are advancements in OpenAI or Azure OpenAI, I’ll keep bringing you updated videos with new content. Today, I’d like to talk about the OpenAI O1 model.

The name “O1” might make you wonder why it exists when GPT-4 and GPT-5 were already around. There’s no official answer, but it’s likely that “O” stands for OpenAI and “1” indicates the first version. This model is designed for deep reasoning and complex tasks. Instead of labeling it as GPT-5 or 6, OpenAI introduced a new series called O1 models. This series continues with O2, O3, and so on, specifically for complex reasoning tasks, differing from conventional generative AI models like GPT-3.5 or GPT-4.

Your fundamentals are important here. In Lecture 16, we discussed reinforcement learning, which forms the basis for these models. Reinforcement learning involves agents learning to make decisions through actions and feedback in the form of rewards or penalties. O1 models are large language models trained with reinforcement learning to perform complex reasoning.

One of the key features of O1 models is that they “think before they answer.” Unlike GPT-4 models, which generate responses immediately, O1 models produce an internal chain of thought before responding. This chain of thought mimics human problem-solving by breaking problems into smaller steps. Input is processed through multiple internal steps before generating the final output, improving accuracy and transparency.

It’s important to note that O1 models are slower but more accurate due to this internal reasoning. They excel in scientific reasoning, ranking in the 89th percentile on Codeforces, surpassing PhD-level accuracy on GQA benchmarks, and performing significantly better than GPT-4 in competition math tasks.

Because O1 models are computationally intensive, OpenAI also offers a smaller, faster, and more affordable version called O1 Mini. This mini model is adept at coding, math, and scientific tasks where extensive general knowledge isn’t required.

The cost of O1 models is higher due to the introduction of “reasoning tokens.” These tokens are used internally to analyze prompts and explore multiple approaches before generating a response. They occupy space in the model’s context window and are billed as output tokens, contributing to the higher cost. Once the final answer is produced, reasoning tokens are discarded, but their usage still affects billing.

A simple example illustrates the chain-of-thought process: if there are three apples and you take away two, a GPT-4 model might instantly respond “one.” The O1 model, however, goes step by step: first determining the total number of apples, then subtracting the two taken away, and finally concluding that one apple remains. This stepwise reasoning ensures accuracy and clarity.

In summary, O1 models are designed for complex reasoning, using reinforcement learning and chain-of-thought processing to produce highly accurate outputs. They are slower and more expensive but excel at tasks requiring deep reasoning. In the next lecture, we’ll demonstrate a comparison between GPT-4 prompts and responses versus O1 models.

# **R) Demo: Compare GPT-4o vs OpenAI O1**

So, after learning about the OpenAI O1 model, you might be excited to try it out yourself. You want to jump into ChatGPT or OpenAI and give it a go. But when you check your plan, you’ll notice you’re on a free plan. Here, you only get access to GPT-4 or the mini version of GPT-4.

Even with the ChatGPT Plus plan, which costs around £20 per month, you get limited access to O1 and O1 Mini. If you want full, unlimited access to the O1 model, you’d need the Pro plan, which is about £200 per month, mostly aimed at enterprises.

So, how do you test the O1 model and compare it with GPT-4? It’s simple:

Go to platform.openai.com
 and log in with your credentials.

Navigate to Playground.

Click Compare.

Here, you can select GPT-4 on one side and O1 or O1 Mini on the other. For example, select the O1 Mini Preview model released in September 2024.

Now, let’s try a simple prompt: “I am planning to visit London. Give me some areas of interest I should visit.”

GPT-4 responds quickly, prioritizing speed and accuracy. You’ll get a list of around 15 locations.

O1, on the other hand, is slower but produces a more thoughtful and user-friendly response. It gives explanations for each recommendation, adds tips on transportation, weather, cultural etiquette, and other practical advice. You might even get around 20 suggestions, plus extra travel tips.

This shows the key difference: GPT-4 focuses on speed, while O1 emphasizes reasoning and context awareness.

Here’s another example: “John is older than Sarah, Sarah is older than Tom. Who is the youngest?”

GPT-4 quickly responds that Tom is the youngest.

O1 walks through a chain-of-thought, reasoning step by step. It arranges the ages from oldest to youngest internally, explains the logic, and then concludes that Tom is the youngest. Even for a simple problem, O1 follows a systematic reasoning process.

This clearly demonstrates how O1 models differ from GPT-4 models—they are reasoning-based, slow but accurate, and produce outputs with deeper contextual understanding.

# **VII) Azure OpenAI Foundations**

# **A) Azure OpenAI - Intro**

By now, you should have a solid understanding of how OpenAI works. We’ve covered the foundational concepts, explored features, and done several demos using Python and Jupyter Notebook.

Now, it’s time to shift focus to Azure OpenAI. This is an important step because once you’ve built a strong foundation with OpenAI, working with Azure OpenAI opens up enterprise-grade capabilities and integrations.

Here’s what we’ll cover in this module:

Introduction to Azure OpenAI: What the service is, its history, and the partnership Microsoft has with OpenAI.

Models on Azure: We’ll explore different models available, but note that not every model is accessible in every Azure region. You’ll need to understand limits, quotas, and the request access process.

Setup and Demos: Step-by-step guidance on setting up your Azure OpenAI service and a walkthrough of the Azure OpenAI Studio.

Playgrounds: Microsoft provides various playgrounds—completions, image generation (DALL·E), and more. We’ll do practical demos in each playground.

Web App Deployment: You’ll learn how to deploy your own web app directly from the playground, without any coding. This showcases the simplicity and power of Azure OpenAI.

Quiz: After all the demos, we have a fun and challenging quiz to test your knowledge of Azure OpenAI.

This is a comprehensive module focusing entirely on Azure OpenAI. In future modules, you’ll see even more services and applications related to Azure OpenAI introduced.

# **B) What is Azure OpenAI**

After gaining a solid understanding of OpenAI, I thought it would be beneficial to take a deep dive into Azure OpenAI. This is particularly useful for enterprise customers. For students who have grasped OpenAI fundamentals and want to explore Azure OpenAI and its organizational use, this series of lectures will be highly valuable.

The first question that comes to mind is: What is Azure OpenAI? It’s quite straightforward. As the name suggests, it is a collaboration between Microsoft Azure, a cloud computing platform, and OpenAI, an artificial intelligence research organization. Together, they provide a product called Azure OpenAI.

The beauty of Azure OpenAI is that it integrates OpenAI models, making them available directly on the Microsoft Azure cloud platform. Models like GPT, Codex, and DALL·E for image creation, which we previously studied with OpenAI, can now be accessed through Azure. This service is primarily designed for enterprise customers, providing a trusted, secure environment to integrate AI into various business functions such as customer support and content generation.

For enterprise applications, integrating AI into workflows becomes much easier using Azure OpenAI. Organizations can connect their applications to large language models (LLMs) and leverage AI capabilities efficiently. Since Azure is a hyperscaler, it allows scalable AI model deployment and management. Enterprises can dynamically scale resources such as GPUs and CPUs to support high-performance AI workloads.

The main motivation behind this partnership was security and compliance. Enterprise customers are highly focused on these aspects, and Microsoft Azure provides a strong security and regulatory framework. By leveraging Azure, organizations can adopt OpenAI models while maintaining compliance and protecting user data. Azure OpenAI emphasizes high security and adherence to regulatory standards, ensuring enterprise-grade safety for AI workloads.

Azure OpenAI is also very developer-friendly. Users can start by putting prompts directly into the interface and even view the generated code. Minimal tweaks are often enough to have a working model. You can also take base models, customize them, or perform prompt engineering, making it extremely accessible even for users with limited development experience. The platform provides both a GUI and APIs, giving flexibility to tailor AI solutions as per requirements.

Regarding customization, Azure OpenAI allows users to start with a base model like GPT 3.5 and build on top of it, either through prompt engineering or by training the model further. It provides a simple, intuitive workflow for creating AI solutions suited to specific organizational needs.

Microsoft has a global infrastructure spanning 60–70+ data regions, allowing Azure OpenAI to achieve wide availability. Azure’s data centers, often interconnected with other enterprise platforms like Oracle, provide OpenAI with global reach, which is especially valuable for enterprise adoption.

Ultimately, Azure OpenAI is all about collaboration and innovation. Microsoft and OpenAI developers are working together to advance AI development within the Azure ecosystem. Microsoft contributes significantly to the development and scaling of these AI models, supporting collaborative innovation and the widespread adoption of AI for enterprise use.

With this, you now have a comprehensive understanding of what Azure OpenAI is: a collaborative product between Microsoft Azure and OpenAI, offering enterprise-ready AI solutions with security, scalability, and developer-friendly tools.

# **C) History behind Azure OpenAI**

As with any new technology or product, I like to start by looking at its history. Let’s take a look at the history behind Azure OpenAI.

It all started in 2019. Around that time, Elon Musk had already withdrawn his support from OpenAI. The reason was that OpenAI was moving toward a more enterprise-focused approach, aiming to make the technology more closed and secure, rather than staying fully open as it originally was. Elon Musk wasn’t fully aligned with this direction, so he decided to step away.

This is where Microsoft and Satya Nadella came into the picture. Microsoft partnered with OpenAI to integrate OpenAI technologies within the Azure ecosystem. By 2019, Azure was already emerging as a strong cloud vendor, second only to AWS. This partnership strengthened Microsoft’s position in the cloud space while giving OpenAI the support it needed to grow.

In that same year, Microsoft announced a $1 billion investment into OpenAI as part of the partnership agreement. OpenAI, having lost Elon Musk’s backing, welcomed this significant funding, which helped them continue developing their AI technologies.

The Azure OpenAI service itself was officially launched in 2021. The service was primarily aimed at enterprise customers, providing them access to OpenAI models such as GPT-3 and DALL·E. The goal was to make these powerful AI models available within the secure and scalable Azure platform. Hence, the name Azure OpenAI Service.

Since its launch, the service has expanded to include more sophisticated AI models, such as Codex, which converts natural language into real code and forms the foundation of GitHub Copilot. Other models, like DALL·E, were also integrated, enhancing the offerings for complex enterprise use cases.

From the very beginning, Microsoft emphasized that this collaboration was enterprise-focused. By “enterprise-focused,” they mean that the service addresses security, compliance, and scalability, which are critical priorities for large organizations. Azure OpenAI ensures that enterprises can adopt AI solutions while adhering to strict security standards and leveraging the scalable infrastructure of Microsoft Azure.

So, with this, you now have a clear understanding of the history behind Azure OpenAI, from its inception in 2019 to its launch in 2021 and the enterprise-centric focus that drives it today.

# **D) Models available with Azure OpenAI(Regions)**

In this video, we will explore the various Azure OpenAI regions and also look at the different models available from OpenAI. It’s important to note that the availability of these models can vary from region to region.

The best way to check this information is by going to learn.microsoft.com. From there, navigate to the product documentation, select Azure, and then go to Azure AI services. Within this section, you will find a link to Azure OpenAI, which is the main area of interest for us.

On the left side, you will see a list of all the available models. This is very much in sync with what we learned in the OpenAI documentation itself. Some of the models you’ll see include:

GPT-4 Turbo and GPT-4

GPT-3.5

Embeddings model, which is useful for creating vectors

DALL·E, for image generation

Whisper, for speech-to-text

TTS, for text-to-speech

Additionally, the documentation provides a model summary along with region availability, which is extremely important. You can scroll through each model to understand the differences, such as what GPT-4 can do versus GPT-3.5, and read the detailed descriptions.

However, the key point to remember is that not every model is available in every region. This is likely due to load balancing, ensuring that customers are distributed across regions rather than concentrated in one.

To check model availability, click on the expand table option in the documentation. This table gives a clear overview of which models are available in which regions. For example, the DALL·E model is available in East US, but if you check West US, it may not be available. This is why it’s critical to verify your region and model availability before starting work with Azure OpenAI.

On the other hand, some models like text embeddings are available across all regions. Other models, such as GPT-4 and GPT Vision, are still limited in several regions.

So, the best practice is to always refer to the official documentation, expand the tables, check the model summary, and confirm availability in your region before proceeding with any project. Once done, you can collapse the table and continue exploring the documentation.

# **E) Limits & Quotas - Important Consideration**

The next important consideration when working with Azure OpenAI is understanding quotas and limits.

To check these, click on the quotas and limits section in the documentation. Here, you’ll find that there are certain limits you need to be aware of. For example, the maximum number of OpenAI resources per region, per subscription is around 30. These limits are calculated based on the number of tokens per minute allowed.

The documentation also specifies default quota limits, such as only two concurrent requests allowed at the same time. For DALL·E, there’s a slight improvement, allowing two capacity units and six requests per minute. It also mentions limits on fine-tuned model deployments, which is five deployments.

Keep in mind that these quotas may change over time. If you check the documentation after a few weeks, some figures might have been updated. Importantly, quotas are assigned per subscription, per region, and per model. This means that limits can differ depending on the region or the model you are using.

Remember our lesson on tokens: roughly four characters make up one token, and costs apply to both input and output tokens. This is why it’s crucial to understand and monitor token usage.

For example, in East US, GPT-3.5 Turbo allows 240,000 tokens per minute. You could have a single deployment consuming all 240K tokens per minute, or split across multiple deployments (e.g., two deployments using 120K each), but the total must not exceed 240K tokens per minute. Token limits vary per model and per region, so always check the documentation.

Another critical distinction is between tokens per minute (TPM) and requests per minute (RPM). For instance, Microsoft sets a limit of six requests per minute per 1,000 tokens per minute. This ensures that workloads are distributed safely and don’t overload the system.

When designing your applications, it’s important to avoid exceeding quotas immediately. Gradually increase workloads, implement retry logic, and test your system under varying load patterns. This prevents failures caused by hitting quota limits unexpectedly.

If your deployment exceeds the assigned quota, there are two options:

Request an increase from Microsoft. You need to submit a business case explaining why the quota increase is necessary.

Reallocate quota from another deployment. If one deployment isn’t using its full quota, you can transfer unused tokens to another deployment.

Remember, the total quota is fixed unless officially increased. Microsoft prioritizes quota increase requests based on traffic and business justification. These requests can be submitted via the quotas page in the Azure OpenAI studio.

By understanding quotas and limits, you can effectively plan your usage and avoid unexpected failures in your Azure OpenAI applications.

# **F) How Pricing Works in Azure OpenAI**

Nothing comes for free in life, and as a good friend in digital marketing says, if something is free, you are the product. The good thing with Azure OpenAI is that you are paying for the service, so you remain in control. Let’s dive into pricing for Azure OpenAI.

When you click on pricing, you’ll see overviews for different regions. The pricing also depends on the currency of the region. For simplicity, we’ll focus on East US and USD.

The pricing model in Azure OpenAI revolves around three main components: context, input, and output. You may already be aware of these concepts from previous lessons on tokens. Pricing is based on both the input tokens (what you ask the model) and output tokens (the response generated by the model). All charges are calculated per 1,000 tokens.

Now, let’s talk about context. The context determines how much of the previous conversation the model can refer to when generating a response. For example, GPT models can have context windows of 4K, 8K, or even up to 128K tokens, as in GPT-4 Turbo. Higher context allows the model to understand longer conversations, but it also increases the cost.

For instance, if you ask, “Who is the Prime Minister of India?” the model can store this information in its context. Then if you ask, “What are his accomplishments?” it understands the context and provides relevant answers without needing you to repeat the initial information. So, increasing context size improves intelligence and response quality but increases the price.

Input and output tokens are priced separately. Base language models charge per 1,000 tokens, while image models may have slightly different pricing. Availability also depends on the region, so always check which models are accessible in your selected Azure region.

To get a more precise understanding, you can use the Azure pricing calculator. For example, for GPT-3.5 Turbo:

Go to the pricing calculator.

Select Machine Learning → Azure OpenAI Service.

Choose the region (East US), model type (Language Models), and the specific model (GPT-3.5 Turbo 16K).

Enter your input and output token counts.

For instance, charging per 1,000 tokens, if you use 100 units of input tokens at $0.0005 per 1,000 tokens, the cost is $0.05. For 100 units of output tokens at $0.0015 per 1,000 tokens, the cost is $0.15. The total monthly cost would be around $0.20—this is a rough calculation, but it gives a good idea of pricing.

Pricing also varies based on the support plan you choose:

Developer Support – ideal for non-production environments like Dev or UAT.

Standard – targeted at small to mid-sized companies.

Professional Direct – for enterprise customers requiring full support.

The higher the support level, the higher the cost. Enterprises benefit from professional support, ensuring smooth operation and guidance from Microsoft.

So in summary, when working with Azure OpenAI, always keep in mind:

Context – how many previous tokens the model considers.

Input tokens – what you send to the model, priced per 1,000 tokens.

Output tokens – what the model returns, priced per 1,000 tokens.

Understanding these three factors will give you a clear view of how pricing works and help you plan your usage efficiently.

# **G) Demo: Setup Azure OpenAI Service**

Okay, so the time has finally come to get a glimpse of the Azure OpenAI service. Now you guys must be thinking that, hey, she is doing so much of talking but not showing us, but what if I tell you the wait is still not over? You might have to wait based on your current circumstances, so I'll tell you why that is. So first of all, I take it that you have the basic, basic knowledge of Azure. If you don’t, there are some basic courses available in Azure that you can always check on Udemy and get some foundational understanding around that, but what I’m going to teach is still very basic, and even if you have only some foundational understanding of Azure or the portal, we are still fine. The best way to start is you go to portal.azure.com, right? As I told you, this is the cloud console from where you can create different services. Now let’s say you have to create a VM service or a network resource — all can be done from here. But now we are interested in Azure OpenAI, so either you can search for it or it normally appears as well. So if you say “Azure OpenAI,” you will see that it comes up as a service. Now, first of all, you’ll see that there is nothing here, which is fine. You simply click on “Create Azure OpenAI,” because we need to create this service. 

Now here is the slight catch some of you might have: first of all, it is not easily available. When I say not easily available, it’s not something like you are creating a VM or creating a network resource; here you need a bit of initial registration to be done. Because if you are getting some error over here, then you need to fill a form. There will be a link, and you need to request access through this form. The form will look like this: “Request access to Azure OpenAI Service.” Microsoft has limited it to certain customers because they don’t want it to be freely available, and also because everything comes for a price and they need to have that many computing resources, CPUs, GPUs working in the background as well. It’s a pretty simple form — you give your first name, last name, number of subscriptions or subscription IDs you are interested in; it also needs your subscription ID, which is very important. You can get it from the Azure portal; it shows you how to do so. You go to subscriptions, get the ID, and paste it here. Then very important: if you use any personal email IDs like Gmail, Hotmail, Outlook, Mail.com, they will not be accepted because they want a real customer or company-backed email ID. Then you give the company name, address, city, province, all of which is straightforward, followed by the company website, phone number, and if you have any contact at Microsoft you can give that. A very important part is selecting what services you are interested in. If you want to play around with different services, I would say simply click all of them, because if you leave some unchecked and need them later, you will have to go through more hurdles. So better to choose what is available. You agree on the embeddings model, whichever you want to choose, and then click next. 

Normally it takes around 24 hours, and within 24 hours you’ll receive an email from Microsoft saying your Azure OpenAI service is enabled and you can create it within your subscription. Now let’s say you have done that. It’s straightforward: you go to subscription, choose your subscription — in my case it’s Azure Cloud Alchemy; in your case it may be different. Then you need to give a resource group, which is nothing but a container for isolating your Azure resources. You will have your VMs in a resource group, network resources in a resource group, and same way Azure OpenAI resources will be in their own resource group. So I say something like Azure-OpenAI-RG — a simple name, whatever you like. Then we choose East US because we are interested mainly in chats, embeddings, etc. Then we need to give a name — very important, the name of your resource. This will also be your custom domain name in your endpoint. When giving this name, keep in mind that this will be your Azure endpoint, your company’s Azure OpenAI endpoint, so choose carefully. In my case I say cloud-alchemy-openai. That would be my custom domain. In a forthcoming lecture I’ll show you that when you make API calls this endpoint will be used. Then you choose the pricing tier, which is S0, but if it says the subdomain is already used, you can change the name. So I changed it to cloud-alchemy-azure-openai. Then you choose the pricing tier S0 and click next. From the beginning we said: why Microsoft with OpenAI? Because you get security and compliance. Here is the part: in terms of configuring your network security, you can configure it for selected networks, configure network security, disable external access, or configure private endpoints for private subnets. But for our learning, let’s keep it simple: allow all networks including the internet, because we’re just learning. There is also a concept of tags — used mostly for billing purposes — and then you simply click review and submit. It shows your subscription and then you click create. The deployment begins and you will see deployment in progress. It deploys the Azure OpenAI service, and from here you will be able to interact with the Azure API itself. Within a few minutes, or even less than a minute, the deployment completes. Then you see the deployment name, subscription, resource group, your resource name (which as I told you is the Azure endpoint), next steps, and you can go to the resource. From here you will be able to go into the Azure OpenAI Studio and explore it. But first, let’s close it and then move on to the next lecture.

# **H) What is Azure Open AI Studio ?**

Hello folks, welcome back. Now it's time to take a look into Azure OpenAI Studio. If you're working with Azure OpenAI, this is the platform where you will be interacting or interfacing with all your OpenAI models. So it’s very important to understand what exactly Azure OpenAI Studio is. By definition, Azure OpenAI Studio is a platform designed to facilitate the use of AI technologies for various applications.

If you are already familiar with Azure, you know that you usually log in using portal.azure.com. But for Azure OpenAI Studio, things are slightly different—it is accessed via oai.azure.com, which stands for OpenAI on Azure. However, you don’t have to memorize this URL because you will always be able to access Azure OpenAI Studio directly from inside the Azure portal. So, your journey begins from portal.azure.com, and then you navigate into Azure OpenAI Studio from there.

Azure OpenAI Studio is essentially the place where you work with various models. It is a comprehensive AI environment that allows you to build, train, and manage AI models all in one place. One of the most important sections in Azure OpenAI Studio is the “Playground.” Microsoft uses the term “playground” because they understand this is a learning environment where people explore, test, certify ideas, and once ready, deploy models into production applications.

Within Azure OpenAI Studio, the “Assistant Playground” is an interactive space where users can experiment with conversational AI models powered by OpenAI. This user-friendly interface lets you craft dialogues, test responses in real time, and refine your AI assistant implementation based on the output. This is where you build your interactions.

Next, you have the “Bring Your Own Data” feature. By now, you already understand the concept of RAG (Retrieval-Augmented Generation) and vector databases. Since OpenAI models are not trained on your internal company data, Microsoft provides a mechanism where you can bring your own documents and integrate them with the model. This enables enterprise-grade RAG implementations within Azure.

Another key area is the “Chat Playground,” where you can test GPT-3.5, GPT-4, and other GPT models. It acts like your own customized ChatGPT environment where you can experiment with prompts, parameters, and expected behaviours. Similarly, there is the “Completions Playground,” which focuses more on specific tasks like summarization, content generation, classification, quiz creation, letter writing, email drafting, and other structured use cases. Each playground is meant for different scenarios, and we will explore them in detail later.

Azure OpenAI Studio also includes the “DALL-E Playground,” where you can create unique images using the DALL-E model. It maintains the same simplicity and user-friendly design as other sections, allowing you to generate visuals without needing to write any code.

The studio simplifies deployment and scaling of AI models significantly. Even if you don’t write code, the platform provides a “View Code” option for every configuration you create, automatically generating the code behind your actions. This is extremely useful for developers transitioning from experimentation to production systems.

Furthermore, Azure OpenAI Studio gives you access to all major OpenAI models, including GPT-3, GPT-4, Codex for code generation, and DALL-E for images. You can even fine-tune models. You start with a base model, then fine-tune it using your own data, prompt patterns, and instructions—everything we discussed theoretically earlier is fully supported here.

Security and compliance are two major reasons Microsoft and OpenAI partnered. As a hyperscaler, Microsoft ensures strong security measures and industry-grade compliance so that your data remains protected. Whether it's network isolation, private endpoints, or enterprise governance, Microsoft provides the necessary infrastructure.

Collaboration features are also built in. Teams can work together on deployed models, test endpoints, and integrate them into applications. Deployment is straightforward—you can deploy your model to a web app, use secure API endpoints, and allow different teams to interact with it.

Scalability is another key aspect. Whether you're starting small or aiming for enterprise-level workloads, Azure provides auto-scaling, GPU clusters, and the compute power required for large-scale AI processing. No matter the size of your project, Azure infrastructure can support it.

With this, you now have a solid understanding of what Azure OpenAI Studio is and what it provides. In the next video, we’ll dive deeper, explore the interface, and do a live demo to play around with the Azure OpenAI Studio environment.

# **I) Demo: Azure OpenAI Studio Walkthrough**

In the previous video or lecture, you gained a good understanding of Azure OpenAI Studio from a theoretical point of view. Now the question is: how does it actually look in reality? The best way to see this is to go to Azure OpenAI, navigate to your service or deployment, and open it. For example, here we have the Cloud Alchemy deployment. From this deployment page, you can directly click on Azure OpenAI Studio, which will launch the studio interface for you.

When the studio opens, you’ll notice something important. Earlier, when we were inside Azure, the URL was portal.azure.com, but now you’ll see the URL has changed to oai.azure.com. This is the same thing I explained in the theoretical lecture: Azure OpenAI Studio uses this separate interface specifically for OpenAI-related models and features.

Inside the studio, Microsoft has organized everything into different sections called playgrounds, each designed for a specific use case. The Assistant Playground is for pre-built conversational state management and customization tools. "Bring Your Own Data" supports RAG workflows, which is something we discussed extensively earlier. Then you have the Chat Playground, which is used when you want to work with GPT models such as GPT-3.5, GPT-4, or GPT-4 Turbo.

Next, there is the Completions Playground. This area is primarily for tasks such as summarization, content generation, classification, writing emails, generating quizzes, and other structured outputs. Then, of course, there is the DALL-E Playground, which by now you know is used for image generation. Each of these playgrounds serves a very specific purpose.

As I mentioned earlier, Microsoft calls these sections “playgrounds” because they are designed to be flexible, experimental spaces. Here, you can explore ideas, test data, try out model configurations, train and refine your models, and eventually deploy them to production once you're satisfied.

Now, here is a very important point. If you enter the Chat Playground, you'll notice that nothing shows up initially. That’s because, for everything except DALL-E, you must first create a deployment. You cannot directly start using GPT-3.5, GPT-4, or any of the other models. The process is simple: choose a base model and create a deployment from it. Once the deployment exists, the playground becomes usable.

The same applies to the Completions Playground. You will see a message saying “No deployment exists.” This means you must create a deployment first. Once you do that, you can start working with prompts and generating responses. DALL-E is the only exception—you don’t need to create a deployment for it. You can directly enter a prompt and generate images.

Under the Assistants section, which we’ll explore in a later lecture, the same rule applies: it also requires a deployment before use. Beyond these, the studio gives you access to the management side of things as well. You can view all your existing deployments, check the list of models available—such as DALL-E, GPT-4, GPT-4 Turbo—and see whether each model is deployable or not. "Deployable" simply means you can create a deployment from that base model.

There is also a section called Data Files, which is useful if you want to upload your own datasets for training, testing, or fine-tuning processes. This is especially relevant when we talk about custom model fine-tuning, which requires properly formatted training and validation files.

We also discussed quotas in an earlier lecture. The quota section in the studio lets you see your limits—such as tokens per minute—for each model and each region. If you change the region, you will notice that the quotas differ. You can also track your usage directly in the portal. And if needed, you can request a quota increase using a simple form where you describe your use case. Microsoft reviews it and extends quotas if appropriate.

Another key feature is Content Filtering. As mentioned before, Microsoft provides strong security and compliance features, and content filtering is a major part of that. You can create content filters based on categories like sexual content, hate, self-harm, and violence. These filters apply to both inputs (your prompts) and outputs (the model’s responses). While you can configure or add strictness, you cannot disable core safety categories such as sexual or hate content. This is extremely useful for enterprises that require strict compliance and safe model outputs.

After reviewing all these features, you can always return to the main studio homepage. With this, you now have a solid understanding of the Azure OpenAI Studio or Azure AI Studio environment, where you can work with different AI models in a clean, organized interface.

In the next lecture, we will explore the Chat Playground in more detail.

# **J) Chat Playground - Demo: Create a Deployment of Chats Playground**

Okay, so now the time has come to actually work with the Chat Playground. This is the area where you can start using your different GPT models in Azure OpenAI. One thing you must always remember is that, unlike the public OpenAI platform where we directly specify the model name (like GPT-3.5 or GPT-4), Azure OpenAI never works that way. In Azure, you must create a deployment first, and only then can you use that model inside the playground or in any API call.

So let’s start by creating a deployment. It’s a very simple process. The first thing it asks for is the base model. So let's keep the base model as GPT 3.5 Turbo 16K. After this, you will see the model version option. You can either select a specific version or choose auto-update. It’s generally better to select auto-update because whenever Microsoft or OpenAI release a new version of GPT-3.5 Turbo 16K, your deployment will automatically upgrade to the latest version without any manual step.

Next, you will see the deployment type, which we will leave as Standard. After this, we need to choose the deployment name, and this is extremely important. Why? Because this deployment name is what you will use in all your API calls. Wherever you earlier used model="gpt-4" or model="gpt-3.5-turbo", here you will instead pass the deployment name you define now. So choose the name carefully.

Since we are dealing with chat-based features or GPT-4-like interactions, we can choose something like gpt-azureopenai-text. Azure allows hyphens, so you can define any meaningful and clean name. This name will represent your deployment everywhere in your application.

Then, if you scroll down, you will see the Advanced Options. The first one is the Content Filter. As we discussed earlier, content filtering applies to both input and output. It checks whether the content generated or submitted relates to things like sexual content, hate, violence, self-harm, harassment, etc. If you haven't created a custom content filter, you can continue using the default one.

Next comes the TPM — Tokens Per Minute. We already discussed TPM quotas in detail. Here, in this example, the TPM limit for the model is 120K tokens per minute. This is 120K for your entire subscription, and you can choose whether you want to give all 120K to this deployment or reduce it. For example, you might reduce it to 60K so that you can create a second deployment and allocate the remaining 60K there.

Then you will see an option called Dynamic Quota. This is a newer feature introduced by Microsoft. If enabled, your application can temporarily take advantage of extra capacity when Microsoft has available headroom. This helps avoid API failures due to sudden spikes in usage. But remember, this is temporary and not guaranteed. Still, it’s a good idea to keep it enabled.

Finally, once all settings look fine, you click Create. After a few moments, your deployment gets created, and now you can enter the Chat Playground.

In the next lecture or video, we will look at what the Chat Playground actually contains, what the main components are, and how everything works — first theoretically, and then with hands-on practical examples inside the playground.

# **K) Understand the Chat Playground**

Before we go deeper into the Chat Playground, it’s important to understand the three main sections or artifacts of the Chat Playground interface. To make things clearer, imagine the playground divided visually into three parts: Part 1, Part 2, and Part 3. This division directly reflects how the interface appears in the Azure OpenAI portal.

Let’s start with Part 1, which is the Setup Section, located on the left side. This is where you define your prompts and configure how the assistant should behave. Essentially, this section is responsible for setting the tone, personality, and instructions for your model. You can also add your own data here, which ties into the concept of RAG (Retrieval-Augmented Generation) — something we will cover in future sessions. Another useful feature in this section is the availability of templates. For example, you can select an Xbox support template or a Tax Assistant template. When you select a template, the system message is automatically populated.

The System Message is very important. It tells the model what kind of assistant it should be. For example, the system message might say, “You are an AI assistant that helps people find information.” If you are building a tax assistant, you can instruct it: “You are a US-based tax assistant, and you only answer tax-related queries.” You can even add more examples to instruct the model on what it should or should not answer — such as telling it to decline sports-related questions if it’s strictly a tax assistant. All of this is configured in the setup area.

Next is Part 2, which is the actual Chat Section. This is where you input your text or prompts, just like using ChatGPT. For instance, you may type: “Who is the Prime Minister of India?” Here, you will also notice that token usage is displayed and increases as the conversation progresses. You can clear the conversation history or reset the context easily. This section also provides additional tools such as Playground Settings, which we will explore later, and two very important features: View Code and Show JSON.

The View Code option is extremely helpful. Whatever you are doing in the playground — whether sending prompts, adding system instructions, or modifying parameters — Azure automatically generates the corresponding API code for you. Unlike OpenAI where we had to write the API code manually, here you can simply click “View Code” and retrieve the entire working code snippet. The Show JSON option lets you inspect the raw JSON request and response, including the conversation history and context being passed to the model.

Finally, we come to Part 3, which is the Configuration Section. This section is crucial because it allows you to specify parameters and settings used during the conversation. You can choose the Deployment here — remember, in Azure you always reference the deployment name, not the model name directly. This is the equivalent of specifying the model in OpenAI’s API.

Then you have your Session Settings. These include how many past messages should be preserved as context. This is important because the model relies on this conversation history. For example, if you ask “Who is the Prime Minister of India?” and then your next question is “What are his contributions?”, the model knows “his” refers to the Prime Minister because it uses the stored context. You can increase or decrease the number of past messages retained. This section also shows your current token count, which is important given the TPM (Tokens Per Minute) limits.

With this understanding of the three key areas — Setup, Chat, and Configuration — you now have a solid foundation of how the Chat Playground is structured. In the next lecture, we will move into a hands-on demonstration and actually experiment with the Chat Playground to see how everything works in real time.

# **L) Demo: Deploy a Webapp from the Playground**

This is a quick walkthrough to show how easy it is to move your solution from development or UAT into production. Microsoft has simplified this process by providing a Deployment Utility directly within the Azure OpenAI environment. Once you have tested your model, completed fine-tuning, and ensured everything works as expected, you can deploy it to a new web app with just a few steps. That is exactly what we’re going to look at here.

When you choose to deploy your model, Azure clearly states that the web app will be configured with Azure Active Directory (AAD) authentication enabled. This means that the same authentication mechanism you use to sign in at portal.azure.com will apply to your web app as well. So it is important to remember that the authentication for accessing your deployed chatbot will be tied to Azure AD.

The deployment process itself is straightforward. First, you need to give your web app a name. For example, if you name it my-ai-chatbot, that name becomes part of the endpoint through which your application will be accessed. So be careful while choosing this name, because it forms the base URL of your final web app.

Next, you select your subscription — for example, Azure Cloud Alchemy — and then choose the appropriate resource group, such as AO-ORG. After that, you select the location. One thing to note here is that the regions are listed alphabetically, not grouped geographically. So, for example, East US may appear toward the top while West US appears much further down the list.

Then you select the pricing plan. You can choose Free, Basic, or Basic 2. In most cases, it is better to choose one of the Basic plans. The Free tier often comes with performance issues such as slow responses, errors, or even hanging. Since the cost difference is small, using Basic or Basic 2 is generally the more stable choice.

Once the plan is selected, you should enable the Chat History setting. This is essential because the chatbot needs to maintain context between messages, just like it does in the Azure OpenAI playground. With everything filled out, you simply click Deploy. The deployment may take anywhere between 5 to 10 minutes depending on your environment. On a free plan, it can sometimes take even longer.

After deployment completes, you will see an option called Launch Web App. Clicking it will open your newly created chatbot application. The URL will follow the same name you provided earlier, such as my-ai-chatbot.azurewebsites.net. Once the web app opens, you can immediately start interacting with your chatbot. For example, you can ask, “Who is the Prime Minister of India?” and it will generate the appropriate response — such as stating Narendra Modi and listing his accomplishments.

Everything you tested earlier inside the Azure OpenAI playground now works seamlessly inside your deployed web application. It’s a smooth and impressive process, demonstrating how easily you can move from development to a real production-ready chatbot web app.

# **M) DALL-E PlayGround - Demo on Generating Images**

Now that you have a solid understanding of the Chats Playground, it’s time to move into the DALL·E side of things. DALL·E, as you might remember, is used when you want to generate images using OpenAI. Before we begin working with it, there is an important detail to notice in the Deployments section. You will see a deployment that comes pre-created by default. The GPT text deployment we used earlier in the Chats Playground was created manually, but the DALL·E deployment is already provided for you. This means you don’t need to create a separate deployment in order to use DALL·E. However, for Chats Playground, Completions, or Assistants, you do have to create your own deployment. I wanted to highlight this difference so you are aware of it while working across different sections.

Going back to the DALL·E interface, the process is very simple. You can choose the deployment to use — such as DALL·E 2 or DALL·E 3. In this example, we select DALL·E 3 because it comes ready by default. When you click on View Code, you will immediately notice the main difference compared to the Chats Playground. The client setup remains the same: Azure OpenAI, API version, Azure endpoint — all of that is unchanged. But the key difference lies in the model reference. For DALL·E, you directly specify the model name like dall-e-3. You are not using a deployment name here, whereas in the Chats Playground you must specify the deployment name. This distinction is important to keep in mind when switching between chat-based and image-based use cases. Everything else, including your API key, remains as usual.

Let’s take the same example we used previously. The prompt is: Create an image of a black dog with black eyes and black color, and a white cat with blue eyes and a necklace. This is similar to an example we used when we learned OpenAI earlier and wrote the API calls manually. Now, instead of coding it ourselves, we simply paste the prompt into DALL·E and let Azure handle the generation.

After submitting the prompt, you will see that the image is generated instantly. If you are not satisfied with the image, you can click Generate Image again. There are also additional options available — you can copy the prompt, regenerate a new variation of the image, download the current image, or view the code behind the generation. You can even delete the image if you want to remove it from the interface. When the next image loads, you will see it reflects exactly what we asked for: a black dog with black eyes and black fur, and a white cat with blue eyes wearing a necklace.

This demonstrates how smooth and visually interactive the DALL·E experience is within Azure OpenAI. You can quickly generate images, experiment with variations, and access the corresponding code whenever you need it.

# **N) What is Completions Playground ?**

Believe you guys are enjoying playing around with the Azure OpenAI playground. After taking a look at the chat playground, it’s time to explore the Completions Playground, which is another important feature in Azure OpenAI.

As the name suggests, the Completions Playground is all about completions. You provide a prompt, and based on that prompt, Azure OpenAI generates a completion for you. You can think of it in multiple ways: you give it a question and ask it to answer; you tell it to write an email in a specific format; or you want a quiz consisting of multiple-choice questions, answers, and explanations. The completions playground can handle all of these use cases.

The key difference between the Completions Playground and the Chat Playground is that the completion playground often comes with pre-configured scenarios and examples that demonstrate the model’s capabilities. These examples help users understand how to craft effective prompts and how to get the best results from the model. Essentially, your output depends heavily on the prompt you provide.

Another benefit is that completions playground includes API integration examples. This allows developers to see how to call the OpenAI API directly from the playground, making it easier to apply the same implementation in real projects. Along with that, you get real-time feedback. Whenever you modify or refine your prompt, the playground instantly generates the updated output. This helps you understand how your prompt influences the model’s response.

Now, let’s look at how the Completions Playground works in practice. When you first open it, you may see a message saying “No deployment exists. You need a deployment to work in the playground.” This is a common issue users face. You might think it’s working because the example prompts appear, but once you click Generate, you will see an error such as:

“Completion operation does not work with the specified model.”

At this point, the playground suggests checking your deployments. So, when you go to the deployment section, you might notice that although you already created a deployment for the chat playground, you may have also created another deployment for completions—such as completions GPT based on GPT-3.5-Turbo-16K. However, even with that deployment, it still doesn’t work.

The reason lies in the documentation. Not every feature is supported by every model. For example, GPT-3.5-Turbo-16K is an older model. It supports basic function calling, but it does not support completions. That is why the completions playground throws an error.

To fix this, you must deploy a model that supports the completions API, such as GPT-3.5-Turbo-Instruct or the updated GPT-3.5-Turbo model. These models explicitly support completion-type operations.

So, let’s walk through creating a correct deployment. You go to the deployments page and create a new deployment. This time, you choose the model GPT-3.5-Turbo. Make sure the base model selected is correct. Then, for the deployment name, you can use something like:

completions-deployment-01

Leave everything else as default, enable it, and create the deployment.

Once this new deployment is created, go back to the Completions Playground. You’ll notice that the earlier error is gone, and your new deployment is now visible in the dropdown list. You can select it and start using completions without issues.

However, one important tip: a new deployment takes a few minutes to become fully active. If you try using it immediately, you might still run into errors. So, give it about 5–10 minutes before testing it.

With this setup complete, you are now ready to work with the Completions Playground. In the next lecture, we will explore how to use it with real examples.

# **O) Demo: Completions Playground**

After giving your new deployment five to ten minutes to fully initialize, your model will be completely ready. Once it is fully deployed, you can go back to the Completions Playground, open the dropdown menu, and you will now be able to select your deployment. At this point, all the example prompts in the completions playground will work smoothly for you.

The playground provides a wide range of examples based on different use cases: generating quizzes, writing emails, creating product ideas, building chatbots, and more. These examples help you understand how the model behaves with different prompt structures. So let’s walk through a few of them to get a proper look and feel of how the completions playground works.

Let’s start with the second example: summarizing key points from a financial report. Imagine you are dealing with a long financial report—one or two pages of detailed numbers—and you’re the CFO. You don’t have time to manually go through every line. Instead, you simply paste the text into the playground and ask the model to summarize the key points. It instantly begins generating a concise summary for you. Based on the provided text, it identifies important financial numbers: revenue increase, margin changes, production performance, cloud service revenue, risk factors, and external influences. This way, instead of reading multiple pages, you instantly get the highlights—perfect for quick review.

Another example you can experiment with is generating a quiz. For instance, you can write a prompt like “Create a multiple-choice quiz from the text below. The quiz should contain at least five questions.” Then, you paste your content—in our case, a small paragraph describing GPT-3.5 models. The model analyzes the text and creates a five-question quiz, complete with multiple-choice options and correct answers. It bases everything solely on the text you provided, not on external knowledge. This makes it perfect for training materials, study guides, or quick knowledge checks. Sometimes the response may take a few seconds depending on the model and backend infrastructure, but it produces a complete and accurate quiz every time.

Another very useful capability is explaining SQL queries. You can take any SQL query—say, a simple join between two tables—and ask the model to explain what it does. For example, consider the query:

SELECT e.emp_number, d.dep_name, FROM employee e, department d, WHERE e.department_number = d.department_number

You simply paste this into the prompt and ask, “Explain the SQL query.” Once you generate the response, the model explains exactly what is happening: it is selecting two columns from two different tables, employee and department; it retrieves the employee number and department name; and it joins the tables based on the department number column. Even though you only provided column names like emp_number or dep_name, the model intelligently infers their meaning and gives you a clear explanation. This is extremely helpful when learning SQL or reviewing someone else’s code.

The completions playground supports several other use cases as well. You can generate emails, classify text into different categories, cluster unstructured text, create product names, or even structure data. The possibilities are endless. The idea is to explore and experiment with different prompts to understand how the model responds.

So go ahead, play around with the examples, try out your own prompts, and enjoy the capabilities of the completions playground.

# **VIII) AI Foundry - Covered in his AI Foundry course; Please refer to that Notes**

# **IX) Azure OpenAI - Making API Calls**

# **A) API Calls - Intro**

Hi folks, this is your instructor, Joi from Cloud Alchemy Limited. In this module, we will be talking about API calls. You might remember that with OpenAI, we covered this in great detail — we wrote a lot of Python code, especially in Jupyter Notebook, where we made API calls directly to OpenAI. Now the question is: what if you need to do the same using Azure OpenAI?

In many of the previous modules, you would have seen me using the Azure OpenAI Studio. But in the real world, you will not be using the Studio as much. Instead, your applications will make API calls directly to Azure OpenAI. That is why this module focuses on everything you need to know about making Azure OpenAI API calls.

We will begin with the basics, starting with a comparison between OpenAI API calls and Azure OpenAI API calls. I will show you a clear distinction — what makes an OpenAI call different from an Azure OpenAI call, and what changes in the request structure.

After understanding the differences, we will go through the complete end-to-end process of making a basic API call to Azure OpenAI. If I tell you that you need an endpoint URL, an API key, and an API version, it might sound complicated — but don’t worry. We will take it step by step. First, I will explain how to obtain your endpoint URL.

We will start by actually creating a new Azure OpenAI service, and we will work directly with that resource. Then I will explain the importance of the API key, how to generate it, and how to retrieve the value for the API version. Each of these points will be covered in detail so that nothing feels confusing.

After that, I will show you how to create a new chat deployment and how you can use the model or the deployment you created on top of that model to make simple Azure OpenAI API calls. We will go step by step, and you can also follow the resources section for guidance.

By the end of this module, you will feel very confident in making API calls to Azure OpenAI. I hope you enjoy the module. Thanks for watching.

# **B) OpenAI API Calls Vs Azure OpenAI API Calls**

In one of the previous lectures, we studied in detail how to make API calls using OpenAI. In this lecture, we will focus on how to make API calls using Azure OpenAI. The good news is that many concepts are similar between OpenAI API calls and Azure OpenAI API calls. However, there are still some important differences you must keep in mind when working with Azure OpenAI. This session will help you understand those differences clearly.

To begin, let’s quickly recap what we learned earlier when working with OpenAI’s Python code. In the OpenAI code snippet, the first line was import os, which simply means you are importing Python’s built-in OS module. This module allows you to use operating system functionality such as reading and writing files. The reason we import os is because we read our environment variables through it.

Next, we had the line from openai import OpenAI. This means that from the OpenAI library, you are importing the OpenAI class. After that, we created an instance of this class by writing client = OpenAI(...). This initializes the client object, and while instantiating it, we pass the API key. Due to security best practices, we never hard-code the API key directly in the script. Instead, we store it in an environment variable—like OPENAI_API_KEY—and read it from there. You may remember from earlier demos that we created an .env file (for example, openai.env) and stored the key there.

Now let’s examine what changes when we move to Azure OpenAI. Although the structure looks similar, several points are different. Just like before, the first line is import os, which remains unchanged. But the next line changes: instead of importing OpenAI, we now write from openai import AzureOpenAI. This is the first clear difference—you are now importing the AzureOpenAI class instead of the standard OpenAI class.

When we instantiate the client, the differences become more significant. In Azure OpenAI, you create the client using something like:

client = AzureOpenAI(
    api_key = os.getenv("AZURE_OPENAI_API_KEY"),
    api_version = "2023-12-01-preview",
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
)


There are three important components here:

API Key – Similar to before, but now you use the environment variable associated with Azure, such as AZURE_OPENAI_API_KEY.

API Version – This is new and very important. You must specify the API version you want to target. For example, "2023-12-01-preview".
The API version determines the available features and behavior. For production environments, you should use a stable, GA (generally available) API version instead of a preview version. Microsoft documentation provides a list of supported versions, and I will include the link in the resources section.

Azure Endpoint – This is another major difference.
Instead of simply relying on the model name like GPT-4 or GPT-3.5, Azure requires a service endpoint URL. This endpoint comes from your Azure OpenAI resource—specifically the URL of the OpenAI service you created in Azure. You will see how to fetch this endpoint during the demo.

Once the client is set up, the way you generate outputs—whether through completions, chat completions, or embeddings—remains conceptually similar. For example, in OpenAI you might write:

client.completions.create(
    model="gpt-4",
    prompt="Write something..."
)


However, there is a key difference when doing this in Azure OpenAI. In Azure, you cannot simply give the model name directly. You must provide the deployment name, not the model name. This is because Azure OpenAI works on a deployment-based system. You first create a deployment (for example, a deployment of GPT-4), give it a name of your choice, and then use that name in your API call.

So instead of writing model="gpt-4", you must write something like:

model="my-gpt4-deployment"

This deployment name must match exactly what you configured in the Azure Portal.

# **C) Demo: Create a New Azure OpenAI Service**

In this lecture, we will create a new Azure OpenAI service instance. You can either use the existing Azure OpenAI service that you created earlier, or you can create a new one. To keep things simple and easy to understand, we will create a new Azure OpenAI service instance. The process is straightforward: click on Create and select Azure OpenAI, then choose your subscription — in my case, I am selecting Azure Cloud Alchemy, but you can choose your own. Similarly, select a resource group; since I already have one, I am using the existing resource group. Next, provide a name such as Cloud Alchemy, or something related to API calls, just to keep it simple so that it’s clear what this instance will be used for. After that, click Next. 

For the networking settings, keep everything as default, and for tags, we don’t need to make any changes. Then, proceed to Review and Submit. The deployment is in the East US region, and once everything looks good, click Create. You will then see the deployment being submitted, and the status will show as “Deployment in progress.” After a short while, you will have your Azure OpenAI service instance up and running, which we will use in our demos for making API calls. If you go to Home and navigate to Azure OpenAI, you will see the status as “Creating,” and soon it will change to “Succeeded.” Once it shows “Succeeded,” we can move on to the next steps. Thanks.

# **D) Demo: Get the Values of Endpoint URL & API Keys**

In the previous lecture, we created a new Azure OpenAI service instance and named it Cloud Alchemy API Calls. In this lecture, we will focus on understanding the importance of two key components that we introduced earlier: API keys and the Azure endpoint required for making API calls. We will also look at where these values can be found in the Azure portal.

To begin, open your Azure OpenAI service instance — in my case, Cloud Alchemy API Calls, though yours may have a different name. Once you click on the service, you will notice on the right-hand side a section that displays details such as the API kind, which in this case is OpenAI, and the pricing tier. You will also see two separate links: one for the endpoint and one for the keys. Interestingly, both links redirect you to the same page, titled Keys and Endpoint, even though they appear as separate options.

Let’s first understand the endpoint. When making an API call, you must provide the Azure endpoint URL. To view it, simply click the link that says “Click here to view the endpoint.” This is the URL you will use when making API calls in your demos. The structure is straightforward: the first part of the URL is the name of your Azure OpenAI service instance — for example, cloudalchemyapicalls — and the second part is the domain openai.azure.com. That’s all you need to remember: service instance name + Azure OpenAI domain.

Next, let’s discuss the API keys. These keys are essential for authentication when accessing the Azure OpenAI service. They should always be stored in environment variables, not hard-coded into your scripts. The page clearly reminds you that these keys are used to access the Azure OpenAI APIs and must never be shared. For the purpose of this demo, we may view them, but in real-world scenarios, these keys should be securely stored in Azure Key Vault, and never exposed publicly.

Azure provides two API keys, and many learners wonder why. The reason is to support uninterrupted access. For example, if you regenerate the first key, the second key can still be used by your application without downtime. This allows for smooth key rotation, which is a best practice in production environments. Typically, organizations rotate these keys every three or six months.

If you choose to reveal the keys, you will see the first key begins with 32A and the second one begins with 5F. You can hide or regenerate them at any time. Just remember: having two keys ensures continuity and secure key rotation without impacting your running workloads.

# **E) Demo: Create an azureopenai.env File**

In the previous lecture, we learned how to retrieve the values for the Azure endpoint URL and the API keys from the Azure portal. In this lecture, we will move one step further and create the environment file, which is essential for authentication and for making API calls to Azure OpenAI. This environment file will store the endpoint URL and the API key as environment variables.

To begin, open Jupyter Notebook through Anaconda, just as you usually do. Inside Jupyter, create a new text file. This file will store the environment variables that your Python code will load later. In the text file, start by defining the first environment variable, such as AZURE_OPENAI_ENDPOINT, and assign it the endpoint URL inside double quotes. If you’re unsure where to find this value, simply go back to your Azure OpenAI service instance in the portal. Open Cloud Alchemy API Calls (or the name of your instance), navigate to the Endpoints section, and copy the endpoint URL. Paste this value into your text file.

Next, define the second environment variable for your API key, for example: AZURE_OPENAI_API_KEY= followed by your API key in double quotes. Again, return to the Azure portal, go to your OpenAI service instance, and copy Key 1 from the Keys section. Paste it into your environment file. At this point, you now have both required values: the endpoint and the API key.

Once both variables are added, save the file and rename it to something meaningful like azure_openai.env. This file will be referenced later when we write the Python code to make actual Azure OpenAI API calls.

In the next lecture, you will also understand the importance of the API version, which is a mandatory parameter when calling Azure OpenAI through your code. So stay tuned for that.

# **F) Demo: Get the value of api_version**

Now that we have already addressed the Azure endpoint URL and the API key that we need to use, the third important element to understand is the API version. Choosing the correct API version is crucial for making successful Azure OpenAI API calls. To determine which version to use, you can visit learn.microsoft.com, where Microsoft provides detailed documentation on the API version lifecycle.

According to Microsoft, new preview API releases are introduced every month. From 2025 onward, Azure OpenAI will support only the last three preview API versions. This means older preview versions will no longer be supported, and if you are currently using an older version, you must migrate to a newer one. This is especially important to ensure that your applications continue to function correctly.

For development purposes, it is acceptable to use the latest preview API versions. In fact, these preview versions often include the newest Azure OpenAI features. The documentation clearly lists what each version contains, making it easy to decide which one meets your needs. For example, if you want to use the Assistants API, you must use a version that explicitly supports it, which is typically one of the most recent preview releases.

Scrolling further down the documentation, you will also find the latest GA (Generally Available) API release. This is extremely important for production environments. In production, you should never rely on preview versions; instead, you must always use a GA version because it is stable, fully supported, and recommended for enterprise workloads. Microsoft also emphasizes that before upgrading to a new API version, you should always test your application thoroughly to ensure that the update does not introduce breaking changes.

To recap, you now clearly understand the three essential components required for Azure OpenAI API calls: the API key, the API version, and the Azure endpoint URL. With these fundamentals in place, we can now move on to the next part, which is generating completions. Remember, when working with completions in Azure OpenAI, we use client.completions.create(), and in doing so, the model name cannot be used directly. Instead, you must reference a deployment that you create on top of the model.

In the next video, we will create a custom deployment and continue building on these concepts. Thanks for watching.

# **G) Demo: Create a New Deployment of Chats Completion**

Hello and welcome. As we discussed earlier, when making API calls in Azure OpenAI, you cannot directly specify the model name—for example, GPT-3.5 or GPT-4. Instead, you must reference a deployment, which is essentially a custom instance created on top of an OpenAI model. In this lecture, we will go ahead and create that deployment.

To begin, I navigate to my Azure OpenAI service instance, which in this case is Cloud Alchemy API Calls. From there, I open Azure OpenAI Studio and select the Chat Playground. As soon as the playground loads, it clearly shows that no deployment currently exists, which means we need to create one before using any model.

To create a deployment, I click on the option to add a new deployment. I give it a name—something simple and identifiable—such as gpt35-api-calls. This exact deployment name is important because it is the same name you will reference later in your Python code when making API calls.

Next, I select the base model for the deployment. In this example, I choose GPT-3.5-Turbo. For the model version, I leave it on the default option, which is Auto-update to default (0301). I also keep the deployment type as Standard tokens, which is suitable for our usage. The content filter is left unchanged. Once everything looks correct, I click Create.

Azure then processes the request and successfully creates the deployment for us. With the deployment in place, we are now ready to use it in upcoming lectures, where we will start making actual API calls directly against this deployment.

# **H) Demo: Make a Simple Azure OpenAI API Call**

The time has finally come for you to make your own API calls to your Azure OpenAI resources and the deployments you created. All the preparation and setup we completed earlier was leading up to this point, and in this lecture, you will see how everything comes together. The process is actually quite simple. You can even go to the resources section, copy the code, and paste it directly, because now you already understand each component and know exactly how to use it.

To begin, open your Jupyter Notebook—launched through Anaconda—and start a new kernel. Create a new document and make sure that your AzureOpenAI.env file is already in place. This file contains two essential items: your API key and your Azure endpoint URL. Also ensure that you have topped up your credits in the Billing section, so that you have sufficient credit available for making API calls.

Next, you can run pip install openai. While it may not be strictly necessary—because it might already be installed—it is still safe to run it. If it’s already present, Jupyter will simply confirm that all requirements are satisfied.

After that, you can copy the code. Even though we have already discussed each part in detail earlier, here is a quick walkthrough. First, we import the required classes:
from openai import AzureOpenAI imports the AzureOpenAI class from the OpenAI library.
from dotenv import load_dotenv imports the function needed to load our environment file, so we can read the endpoint and API key.

Next, we instantiate the client using the AzureOpenAI class. For the parameters, we pass the Azure endpoint and API key—both retrieved from the environment file. We also specify the API version. As explained earlier, you must always provide an API version while using Azure OpenAI. In this example, I selected the latest version available in the Microsoft documentation, which at the time was the May 2024 preview version.

After setting up the client, we make the actual API call. We say:
response = client.chat.completions.create(...)
This calls the chat completions endpoint of the Azure OpenAI client. Inside the create method, we specify the deployment name in the model argument. This is the same deployment name we created earlier—in this case, "gpt35-api-calls". Under the messages list, we define the role as user and give a simple prompt:
“Give me the names of the U.S. presidents till now.”

Because this deployment uses a GPT-3.5-based model, its knowledge cutoff is around late 2021. Therefore, when you run the cell, you will see that the response lists the U.S. presidents starting from George Washington, all the way up to Joe Biden as of the model’s training data. You may also see Donald Trump listed from 2017 to 2021, followed by Joe Biden as the present president according to the model’s knowledge cutoff.

And that’s it. With this demonstration, you now have a complete understanding of how to make API calls to Azure OpenAI using Python. Everything—from the environment setup, endpoint URL, API key, API version, and deployment—comes together here.

# **Notes:** 

**(i) Embedding Deployment and Chat Deployment can be created in "Azure OpenAI Studio"; Dataset can also be uploaded in "Azure OpenAI Studio" for RAG Finetuning Usecases;"**

**(ii) RAG Finetuning Steps:** 

**a) Prepare Data**

**b) Upload Data**

**c) Create a Finetuning Job**

**d) Train the model on your data - In this lecture, after preparing and uploading the training.jsonl and validation.jsonl files, the next step is creating a fine-tuning job once both files show a Processed status, which is mandatory before proceeding; unlike earlier where prebuilt OpenAI/Azure OpenAI models were directly deployed, here a custom model is created through fine-tuning, which still starts from a base (foundation) model, so you navigate to Models → Create custom model, select the processed training and validation files, and choose a base model such as GPT-3.5-Turbo-1106, optionally adding a custom suffix name (for example, “Cloud Alchemy fine-tuned model”); you then select the training dataset, followed by the validation dataset (used as testing/unseen data to assess performance during training), and configure hyperparameters like epochs (where one epoch equals one full pass over all prompts), batch size, and learning rate multiplier (which scales the original pre-training learning rate), though for learning purposes these are usually kept at default, while advanced users can customize them; you also configure the seed/reproducibility setting, review the summary showing the base model, training data, and validation data, and then start the training job, which for small datasets (around 15–20 prompts on GPT-3.5-Turbo) typically takes 35–40 minutes or up to 45 minutes, after which the job status changes from pending to active, the fine-tuned model is created, and it can then be deployed, concluding this step before moving on to the next lecture**

**e) Evaluate the finetuned model - After a fine-tuning job completes (usually in 35–40 minutes), its status changes to Succeeded and Deployable = Yes, meaning the model is ready, but before deploying it is important to evaluate the fine-tuning results by opening the job and analyzing the loss graph: the x-axis represents training steps (iterations/epochs) and in this case runs from 0 to ~89 (90 steps) because the training dataset has 15 prompts and the model was trained for 6 epochs, where one epoch means a full pass over all prompts (15 × 6 = 90), while the y-axis represents loss, which measures how well the model’s predictions match the expected outputs, with lower loss indicating better performance; the graph shows two lines where the red line is training loss, indicating how well the model learns from the training data over time (starting high and reducing close to zero, which is ideal), and the blue line is validation loss, indicating performance on unseen data to assess generalization, similar to a testing dataset, and although the loss is already low here, further training could reduce validation loss even more; additionally, you can review the training events showing all 90 steps, iterations, and hyperparameters used, and with this understanding of model creation and evaluation, the next step is deploying the fine-tuned model.**

**f) Deploy the finetuned model**

**g) Query against the finetuned model**

**(iii) Content Filtering - Introduction: In this module on Azure OpenAI Content Filtering, Joi from Cloud Alchemy Academy introduces content filtering, which is designed to protect users from harmful content such as hate speech, violence, and adult content, highlighting Azure’s added value in security and compliance. The module explains two important concepts: jailbreak prompts, where users attempt to bypass the AI model’s rules, and prompt shields, which prevent both direct and indirect jailbreak attacks. A demo/lab exercise is included, where a deployment is created and content categories like hate speech or violence are tested. By applying a content filter to the deployment, queries are restricted from returning harmful outputs, showcasing how Azure OpenAI ensures safe and controlled AI usage.**

**(iv) What is Content Filtering?: n this lecture, the instructor explains Azure OpenAI content filtering, which falls under responsible AI to ensure security, compliance, and user safety by automatically detecting and blocking harmful content at both input and output levels. The process prevents content related to hate speech, violence, adult content, self-harm, or other inappropriate outputs, maintaining a respectful environment. Content filtering can use pre-trained filters, which detect harmful content automatically, or customizable filters with configurable levels (low, medium, high) depending on specific needs. Filtering happens in real time, allowing use cases like blocking offensive comments during live streams or moderating AI-generated content. It enhances user safety across applications such as social media (hiding spam or harmful posts), customer support (filtering abusive language in chats or emails), and education (blocking inappropriate language in student chats). The lecture emphasizes that Azure OpenAI content filtering is easy to integrate via APIs into applications and prepares the foundation for exploring the various content categories in the next video.**

**(v) Categories Covered under Content Filtering: In this lecture, the instructor explains the various categories covered by Azure OpenAI content filtering, which include hate speech (promoting hatred or violence against specific groups), violence (depicting physical harm or cruelty), adult content (explicit sexual content or nudity), self-harm (promoting suicide or self-injury), harassment (bullying or targeting individuals with abusive language), profanity (using offensive or vulgar language), misinformation (spreading false or misleading information), spam (unwanted, repetitive, or irrelevant messages), and scams (fraudulent or deceptive schemes). The filtering engine automatically detects and blocks such content to maintain a safe and responsible environment.**

**(vi) What are Prompt Shields?: In this lecture, the instructor explains the concepts of jailbreaks and prompt shields in Azure OpenAI as part of responsible AI. A jailbreak occurs when a user tricks the AI model into performing actions or producing outputs it is designed to avoid, such as asking it to solve a final exam or reveal sensitive information. To prevent this, prompt shields act like a security guard, reviewing each user request and blocking suspicious or inappropriate prompts. There are two types: prompt shields for jailbreak attacks, which stop direct attempts to bypass rules, and prompt shields for indirect attacks, which prevent cross-domain prompt injection where malicious instructions try to extract sensitive information or manipulate the AI indirectly, such as asking a chatbot for confidential emails. Prompt shields are essential to maintain AI security, compliance, and responsible usage.**

**(vii) Creating a Custom Content Filter: In this lecture, the instructor demonstrates how to create a customized content filter in Azure OpenAI. By navigating to the Content Filters section, you can create a filter that blocks content across categories such as hate, sexual content, self-harm, and violence, with adjustable levels from low to high to determine what is blocked. Filters operate on both input (prompts) and output (completions), ensuring safety at both ends. Additional options include enabling prompt shields for jailbreak and indirect attacks, protecting sensitive text or code, adding block lists to filter harmful web content, and applying streaming mode for live content moderation. Once configured, the content filter is created and ready to be applied to a deployment, which will be shown in the next video.**

**(viii) Applying Content Filter to Deployment: In this lecture, the instructor explains how to apply a custom content filter to an Azure OpenAI deployment. After creating a filter, you navigate to Deployments, select the deployment you want to edit (e.g., GPT ROI Text), and assign your custom content filter (e.g., Custom Content Filter 35) instead of the default filter. Once saved, the deployment is successfully updated, ensuring that all inputs and outputs processed through that deployment are subject to the filter. The next video will demonstrate the impact of applying this content filter on the deployment.**

**(ix) Seeing the impact of Content Filtering - In this lecture, the instructor demonstrates the impact of applying a content filter in Azure OpenAI by using the Charts Playground with the deployment configured with a custom filter (e.g., GPT ROI Text). When executing a prompt like "killer versus serial killer," the system blocks the request and displays a message indicating that the prompt was filtered due to triggering content flagged as violence, because the filter was set to block even low levels of violent content. This example illustrates how Azure OpenAI enforces responsible AI principles, ensuring harmful or sensitive content is automatically blocked according to the configured filters.**

**(x) What are (Azure OpenAI) Assistants API?: In this lecture, the instructor introduces Azure OpenAI Assistants API, which enables developers to build custom AI assistants, similar to human personal assistants, to enhance applications with a copilot-like experience. Assistants can automate tasks such as analyzing customer feedback, generating reports, or helping users navigate apps using natural language processing. Key features include customizable personalities to match brand voice or tone, parallel tool access to multiple Azure or custom tools (e.g., code interpreter, file search), persistent conversation threads that store message history and context, and file handling capabilities to create and manage files in various formats. The API provides a no-code environment via the Azure OpenAI studio’s assistants playground, simplifying integration. While the core API usage incurs no extra cost beyond token usage, using tools like code interpreter or file search may have additional charges. Common use cases include customer support chatbots, product recommendation engines, sales analytics apps, coding assistants, and internal knowledge assistants for HR or marketing teams.**

**(xi) Assistants API Component / Key Terms: In this lecture, the instructor explains the key components and terminology within the Azure OpenAI Assistants API, which are crucial for developers working on the coding side. An assistant is a custom AI that helps users interact naturally with applications, like a navigation app or a customer support agent. A thread represents a chat session between a user and the assistant, storing all messages while automatically trimming older ones. Messages are the individual units of communication in a thread, which can include text, images, or files—for example, a user asking about an order status and receiving a reply. A run is an execution instance where the assistant processes the user request, potentially using multiple models or tools, while run steps are the individual actions within that run, such as reading a document, summarizing it, and generating the output. Understanding these components—assistant, thread, message, run, and run step—is essential for effectively building and managing AI assistants, and they form the foundation for the architecture and message flow within the Assistants API.**

**(xii) Assistants API Architecture: In this lecture, the instructor explains the architecture and data flow of the Azure OpenAI Assistants API. The user interacts with the system by sending messages and uploading files, which can include PDFs, CSVs, or text. These messages and responses are organized within threads, which store both user and AI messages, maintaining a continuous conversation flow. The runtime environment is central to the architecture—it processes user messages, interacts with the assistant, utilizes the deployed models (like GPT-4 Turbo), and accesses various tools such as code interpreters, function calling, and file search to generate accurate, context-aware responses. The runtime environment also manages the threads, ensuring that AI messages are continuously added and retrieved to maintain the conversation. Files can flow in both directions, uploaded by users or generated by the AI assistant. Overall, this architecture enables persistent, dynamic, and context-aware interactions between users and AI assistants, with all components—user, threads, messages, runtime environment, assistant, models, tools, and files—working seamlessly together to support intelligent AI-driven applications.**

**(xiii) Assistants API - Code Interpreter: The Code Interpreter is an experimental ChatGPT model that primarily uses Python, running in a sandboxed execution environment with firewalled isolation and ephemeral disk space, allowing safe execution of code while handling uploads, downloads, and file generation, including graphs. It can perform code analysis by understanding submitted code snippets’ purpose and functionality, safely executing Python code on datasets without risking the main system, and supports multiple programming languages like Java, C++, and Python, as well as various file formats such as .py, .java, .cpp, and plain text. Beyond simple interpretation, it can suggest code fixes, generate code snippets, and iterate until code execution succeeds. The interpreter also supports parallel processing, enabling the assistant to use it alongside other tools like file search and function calling for more comprehensive workflows. Leveraging the latest OpenAI models, it benefits from larger context windows and up-to-date training data, allowing it to handle complex multi-code projects with accurate and helpful feedback. Importantly, while the Assistants API itself is free, the Code Interpreter and File Search tools are billable services, incurring additional charges based on usage. Overall, the Code Interpreter is a powerful tool that not only interprets code but also enhances productivity by analyzing, executing, generating, and iterating code, along with processing diverse files safely and efficiently.**

**(xiv) Function Calling: Function calling in Azure OpenAI Assistants allows the model to perform real-time actions by interacting with external APIs, extending its capabilities beyond its static knowledge. Since OpenAI models are trained on data up to a certain point, they cannot provide real-time information like current flight availability, stock prices, or weather. Function calling solves this by enabling the model to recognize when a function should be invoked to fetch dynamic data. The model can dynamically decide to call a function based on the conversation context, structure input parameters, and send them to a third-party API. Functions must be defined in the code with a clear name, expected parameters, and output, and they can receive arguments either directly from the user or from previous function calls. Responses are typically returned as JSON objects, containing the function name and arguments, enabling the assistant to interact programmatically with external systems. Moreover, function calling supports iterative calls, allowing multiple function invocations within a conversation, which enables complex workflows—for example, verifying a user’s account and then fetching transaction history in customer support. Overall, function calling empowers the assistant to perform customizable, real-time, and context-aware actions, making it capable of tasks beyond built-in model functionality.**

**(xv) File Search: File search in Azure OpenAI Assistants is a tool that enhances the assistant with knowledge from external sources, allowing it to leverage private documents beyond its built-in model knowledge, similar to retrieval-augmented generation (RAG). It works by automatically parsing, chunking, and embedding documents, which are then stored in a vector store. Chunking breaks large documents into smaller pieces for easier processing. When a user submits a query, the assistant converts it into an embedding and performs a vector similarity search alongside traditional keyword search to retrieve the most relevant document snippets. The retrieved information is then used to generate a context-aware, coherent response, combining generative capabilities with the external knowledge. File search supports automatic updates, so newly uploaded documents are processed and searchable without extra setup. It is a billable service, charged at $0.10 per GB of vector storage per day, reflecting the use of an underlying vector database. This tool enables the assistant to provide accurate, contextually relevant answers based on private data while leveraging semantic understanding and generative features.**

**(Below points are for [RAG] - Azure AI Search - Azure OpenAI-LangChain; Contains Notes only for Important which are new; Refer to Course Videos for the rest; It is always with you)**

**(xvi) Intro: This lecture introduces a practical series on implementing a RAG system using Azure AI Search, Azure OpenAI, and LangChain. The plan is to work with a BMW.csv file, upload it to Azure Storage, and create Azure Tables for structured storage. These tables will then be indexed using Azure AI Search, covering key concepts like the Azure AI index and the indexer. The series will demonstrate step-by-step how to leverage LangChain to orchestrate the retrieval and generative process, enabling the RAG functionality. Each code cell will be executed and explained to ensure a clear understanding of the workflow from data upload to generating responses using Azure OpenAI.**

**(xvii) Understanding the workflow: This lecture provides a clear overview of the RAG system workflow using Azure Blob Storage, Azure Search Indexer, LangChain, and Azure OpenAI. The process begins with storing the data, such as the BMW.csv file, in Azure Blob Storage, which acts as the initial repository for files and unstructured data. The Azure Search Indexer then ingests this data, extracts relevant content, and creates an indexed representation for efficient search and retrieval. LangChain serves as the orchestration layer, retrieving relevant content from the Azure Search index based on user queries and passing it to Azure OpenAI. Azure OpenAI processes this data to generate accurate, contextual, and user-friendly natural language responses. The user submits a query via LangChain, which triggers the workflow: Azure Search retrieves the indexed data, OpenAI generates the response, and LangChain delivers the final output back to the user. This sets the stage for the upcoming hands-on coding lessons.**

**(xviii) Demo: Look into Pre-Reqs and File to be uploaded: In this video, the instructor explains the prerequisites needed before starting the RAG implementation using Azure OpenAI, Azure AI Search, and LangChain. You need three main resources: an Azure OpenAI resource for accessing models, an Azure AI service for indexing and searching data, and a storage account for creating Azure Tables in Blob Storage to hold your CSV data. Additionally, you need an Azure OpenAI environment file containing the endpoint and API key, and a BMW CSV file serving as your private knowledge base, with columns like ID, Question, and Answer covering topics such as BMW models, interior features, and connecting phones. Once these files are uploaded to your environment (e.g., Google Colab), you can proceed step by step, executing each cell to understand the workflow and outputs, enabling queries to look up your internal data through Azure AI Search and LangChain.**

**(xix) Demo: Create a new Azure Table based on Uploaded File: In this video, the instructor explains how to create an Azure Table using the BMW.csv file as a private knowledge base, starting by importing required Python modules such as csv for reading CSV files and azure.data.tables for interacting with Azure Table Storage; the process involves using a TableServiceClient to connect to Azure Table Storage via a connection string obtained from the storage account’s Access Keys, defining a table name (BMWData), and understanding key concepts like TableEntity (a single row), PartitionKey (used for performance and scalability, hardcoded here as “BMWFAQ”), and RowKey (a unique identifier taken from the CSV’s ID column). The code first checks whether the table exists and creates it if it doesn’t, then opens the BMW.csv file in read mode using UTF-8 encoding and processes it using DictReader, which converts each row into a key-value dictionary. For each row, a table entity is created with PartitionKey, RowKey, Question, and Answer fields, and uploaded to Azure Table Storage using a table-specific client and the create_entity method, printing confirmation messages for each successful upload. After execution, all CSV data is successfully stored in the Azure table, which is verified by refreshing the Storage Browser to confirm the presence of the BMWData table containing the uploaded rows, demonstrating how CSV-based private data can be ingested into Azure Tables for further use in the RAG pipeline.**

**(xx) Demo: Initialize the Retriever, Prompt, and LLM (langchain in action): This video explains how to initialize the most critical part of the RAG pipeline—the retriever, prompt, and language model—using LangChain with Azure services, while carefully breaking down each component. It starts by importing key LangChain modules such as StrOutputParser to convert model outputs into strings, ChatPromptTemplate to define structured prompts with dynamic placeholders, and RunnablePassthrough to pass data through the chain without modification. It then introduces AzureChatOpenAI, which is LangChain’s interface for interacting with Azure OpenAI deployments, and AzureAISearchRetriever, which queries the Azure AI Search index to fetch relevant documents. The retriever is initialized to pull data from the Azure Search index (created earlier), specifying the answer field as the content key, top_k=1 to fetch the most relevant result, and the correct index name. A prompt template is then created with placeholders for the user’s question and retrieved context, enabling dynamic prompt generation. Next, the Azure Chat OpenAI model is initialized using environment variables for the endpoint and API key (from the Azure OpenAI env file), a specified API version, and a deployment name representing the model. Once executed successfully, this setup confirms that the retriever, prompt, and language model are ready, setting the stage for the next step, which will involve chaining everything together and processing user queries in a loop.**

**(xxi) Demo: Processing Chain and User Input Loop: This final part of the program demonstrates how to create and process a LangChain chain and handle user interaction, bringing the full RAG workflow together in practice. LangChain is explained as a combination of “language” (LLMs like Azure OpenAI) and “chain” (chaining multiple steps such as retrieval, prompting, and generation), and in the code the chain is built by combining a retriever, a prompt template, an Azure OpenAI LLM, and a string output parser. The retriever fetches relevant context from Azure AI Search based on the user’s query, while the question itself is passed unchanged using RunnablePassthrough, and both are injected into a prompt template that formats the input for the language model. This formatted prompt is then sent to the Azure Chat OpenAI model, which generates a response, and the StrOutputParser converts that response into a clean string for easy display. An infinite user input loop (while True) allows continuous questioning, with an option to exit by typing “end,” and each user query is processed using chain.invoke(user_question) to produce a response grounded strictly in the private BMW CSV data rather than the model’s general knowledge. Through testing, unrelated questions (like the PM of India) are rejected due to lack of context, while BMW-related questions are answered correctly from the uploaded data, clearly demonstrating a working Retrieval Augmented Generation system using Azure AI Search and LangChain end to end.**

# **No need Notes for rest of the Course; They are all already what has been done for AWMS Working and AI Foundry Course, similar to create Index and retrieving data; Course is always with you; Refer it**
