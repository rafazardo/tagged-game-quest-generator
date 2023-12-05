# Tagged Game Quest Generator Using Fine Tuning GPT2 Model

Within this repository, you'll find the comprehensive source code dedicated to fine-tuning the GPT-2 neural network. Our focus is on leveraging the [`TVGQS`](https://github.com/rafazardo/tvgqs) dataset to specifically generate tagged game quests. This process involves training the model to understand and produce quests within the gaming context, enriching the gaming experience for players.

Moreover, our repository doesn't just stop at fine-tuning; it also facilitates the execution of sentence inferences using a pre-trained model furnished by our team. This allows users to swiftly generate and explore diverse quest scenarios, enhancing the adaptability and creativity within gaming narratives.

Developers, enthusiasts, and gaming aficionados alike can dive into the codebase, experiment with the models, and potentially expand upon this groundwork, fostering innovation in quest generation for gaming environments. Whether it's refining existing quests, crafting entirely new adventures, or exploring different gaming genres, the tools provided here offer a gateway to imaginative and immersive quest development.

# Examples of Generated Sentences

Below are a few examples of sentences generated using the trained model we've made available for you below. These sentences showcase the model's capabilities and the kind of output it produces based on the training data

|Sentence|
|--------|
|`<SOS>`I woke late this morning and left my [ITEM] in the [PLACE] before going on duty. <SOA>Could you find my [ITEM]?<EOA> I misplaced it while I was trying to go back to my [PLACE] outpost in [PLACE]. I'll have it back when I get back.`<EOS>`|
|`<SOS>`I am [CHARACTER] Ghastkill, mayor of [PLACE]. I need someone to descend into the mines southeast of town and break the [ENEMY] curse that has been plaguing the nearby miners for years. With your help I can cleanse the tunnels and bolster the town's economy<EOA>.`<EOS>`|
|`<SOS>`I know things. Hidden places, dangerous beasts. All of this to my surprise, they've all been cooked by someone I've never met. I know how to cook [ITEM], but I can't bring myself to cook [ITEM] for everyone. I've never even cooked [ITEM] myself. I've never even heard of [ITEM]. There are so many different [ITEM] to choose from! I know [ITEM] can be cooked in [ITEM], but I need to know how to cook [ITEM] for everyone!`<EOS>`|

# Dependencies

To effortlessly install all the necessary dependencies, simply execute the following code below.

```python
pip install -r requirements.txt -U
```

# Reproducing Results 

Before conducting inferences, it's essential to download the trained model to run the pipeline. To acquire the model, simply access this link [`Trained-TGQG-Model`](). Once the model is downloaded, you can execute the following command to generate the results.

```python
python3 inference.py
  --path_outdir_sentences `Destination path to save the generated sentences`
  --path_indir_trained_model `Destination get the trained model`
  --num_sentences `Number of sentences to be generated`
```

The generated quest results will be stored in a .txt file within the directory specified as a parameter for path_indir_trained_model.
