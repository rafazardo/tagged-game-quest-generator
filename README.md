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

Before conducting inferences, it's essential to download the trained model to run the pipeline. To acquire the model, simply access this link [`Trained-TGQG-Model`](https://drive.google.com/drive/folders/1eUus915kpMYiL7AfZidPHbXa8RZYd4_3?usp=drive_link). Once the model is downloaded, you can execute the following command to generate the results.

```python
python3 inference.py
  --path_outdir_sentences `Destination path to save the generated sentences`
  --path_indir_trained_model `Destination get the trained model`
  --num_sentences `Number of sentences to be generated`
```

The generated quest results will be stored in a .txt file within the directory specified as a parameter for path_indir_trained_model.

# Training the Models

You also have the option to train the model using our dataset, granting you complete freedom to experiment with new training parameters! This opportunity empowers you to delve into training the model using our dataset, providing the flexibility to explore and test various training configurations. It's an open invitation to customize and fine-tune the model according to your preferences and specific requirements. Should you need guidance or support during this training process, feel free to reach out for assistance!

## Dataset Download 

You can download the TVGQS Dataset using the following command: 

```python
python3 download_dataset.py
  --path_outdir_dataset `Destination path to save the dataset`
```

This command facilitates the download of the TVGQS Dataset, ensuring easy access to the dataset for your experimentation and model training purposes.

## Train the models

Now that you have all the necessary resources at your disposal, it's time to set the parameters and utilize the dataset to train the model. 

```python
python3 train.py
  --path_outdir_trained_model `Destination path to save the trained model`
  --num_train_epochs `Number of train epochs`
  --path_indir_dataset `Destination path to load the dataset`
  --path_outdir_train_dataset `Destination path to save the train dataset`
  --path_outdir_test_dataset `Destination path to save the teste dataset`
```
With the required resources in place, you're equipped to define the parameters and employ the dataset effectively to initiate the model training process. This step marks the beginning of an exciting journey toward training models tailored to your specific objectives and requirements. If you need any assistance in parameter selection or dataset utilization strategies, feel free to ask for guidance!

# Citing this Work

If you use our TGQG model in your research, please cite:

```
@misc {zardo_fiorio_feijo_2023,
  title={Tagged Game Quest Generator}
  author={Zardo, Rafael; Fiorio, Pedro and dos Santos, André Luiz F.},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/rafazardo/tagged-game-quest-generator}}
}
```
