
start = 'Taylor Swift\'s artistic taste is characterized by her experimentation with different genres and her focus on pastel colors, vintage fashion, and whimsical imagery. Her attention to detail and unique visual aesthetic has helped her to stand out in the music industry.'
end = 'My research investigates the future of knowledge representation and creative work aided by machine understanding of language. I prototype software interfaces that help us become clearer thinkers and more prolific dreamers.'

start_embedding = ac.embed(start)
end_embedding = ac.embed(end)

for t in torch.linspace(0, 1, 10):
    latent = slerp(start_embedding, end_embedding, t)
    pp(ac.generate_from_latent(latent))



positive_sentences = [
    """Taylor Swift is my favorite artist. She always writes wonderful songs about the best parts of being human -- love.""",
    """This has been a glorious evening, and we should be grateful for our happiness.""",
    """The flowers that decorated the living room of the castle were all so beautiful, it filled us with joy.""",
    """As I walked out of the hospital, I felt relieved and happy that my son was okay.""",
    """I was so proud of myself for going the whole day without crying.""",
    """I have always imagined that Paradise will be a kind of library. The light pouring through the windowed ceiling became brighter, so bright I could see nothing but whiteness.""",
    """It was a bright and sunny day.""",
    """We were doing great -- everything was going according to plan.""",
    """The cocktail we had found was delicious.""",
    """At Apple, we strive to build the best products we can, and deliver the best experiences to our customers.""",
    """This year, The Verge celebrates our tenth year anniversary as an independent publication!""",
    """Hoppy was astonished and grateful for the tidings. As the pair made dinner, tasty odors wafted into the kitchen.""",
    """He seemed to be satisfied, grin on his face.""",
    """The atmosphere was very serene as the sun went down and greeted the evening.""",
    """Let us not spoil this happy occasion!""",
    """We are very glad to see you on deck," said the captain.""",
    """On this occasion, we have much to celebrate.""",
    """A little over a year ago, I visited New York City with a few of my friends, and one of the most memorable places, oddly enough, was a small chess store.""",
    """In that spirit, this year, I'm enjoying the novelties in front of me, and the clarity of purpose around me. I'm trying to make the most of both. There's no hurry. Today, there's much to see, and tomorrow, the fog will lift.""",
    """It's a pretty relaxed Sunday afternoon, and I'm sitting in my office chair in a quiet room instead of lying with my back against my pillow and my feet on my bed in my room. And let me tell you, I miss my bedroom dearly.""",
    """When I got tired from paddling and pushing (which was often during my first week in the water), I loved to just sit on the board and watch the sun inch down over the horizon. This was my favorite time to be out in the water.""",
    """It is rare for a thing to be described purely for what it is, undecorated by what we could easily confuse it to be while carrying on our distracted lives.""",
    """Love is sacred; love is happiness.""",
    """When he came home, he would always begin his evening by singing along to the radio.""",
    """The flowers had bloomed in the garden, dressing the entire neighborhood in a waterfall of vibrant color and haze.""",
]
negative_sentences = [
    """Taylor Swift is my least favorite artist. She never writes any good songs, and I'm just sick of her break-up songs, which is the only thing she ever writes about.""",
    """This has been a sad, gloomy evening, and there is little to be thankful for.""",
    """The dead wilting flowers in the living room in our apartment looked so depressing.""",
    """After I sprinted out of the building, I cried and cried about my dead son.""",
    """I cried every few hours for the whole day, and could never smile. I just couldn't hold it back.""",
    """I have always imagined that hell would be like prison. The light piercing through the windowed ceiling became so intense, it blinded me quickly.""",
    """It was a dim and gloomy day, raining all day.""",
    """We were doing terribly -- nothing was going according to plan.""",
    """The beers we stumbled upon were disgusting.""",
    """At Samsung, we try to build the worst products we can, and ship the worst experiences to our users.""",
    """This year, The Verge collapses as our tenth year approaches, and we have to succumb to an acquisition.""",
    """Hoppy was gravely disappointed in the offerings. As the pair made supper, the smell spread all throughout the house.""",
    """He appeared dissatisfied, tears streaming down his tired face.""",
    """The vibe was chaotic and loud as the sun came up and another day started begrudgingly.""",
    """Let's just move on quickly past this sad occasion.""",
    """We are just shocked and sad to see you back on deck," muttered the captain.""",
    """On this occasion, we have a lot to mourn.""",
    """A little over a year ago, I went to New York with a few of my relatives, and one of the most dangerous places was a dark, dimly lit corner of the park.""",
    """With that in mind, this year, I'm ignoring all the problems behind me, and the mess and confusion of my life around me. I'm trying to just move on past everything. I'm in a rush. Today, there's so much to do, and tomorrow will be worse.""",
    """It's a busy Monday night, and I'm crouched in my office chair in my room instead of lying with my back against the wall, missing everyone I lost.""",
    """When I got tired from paddling and pushing (which was often during my first week in the grind), I got so sad about all the things I couldn't achieve, and just stared at the sun as I dreaded the most boring part of my day.""",
    """Love is sorrow; love is nothing but pain.""",
    """When he came home, he would always just fall asleep, suffering from fatigue.""",
    """The flowers had wilted and died in the window, filling the rest of the neighborhood with sadness and melancholy.""",
]


positive_embeddings = [autoencoder.embed(s) for s in tqdm(positive_sentences)]
negative_embeddings = [autoencoder.embed(s) for s in tqdm(negative_sentences)]



mean_positive_embedding = torch.mean(torch.stack(positive_embeddings), dim=0)
mean_negative_embedding = torch.mean(torch.stack(negative_embeddings), dim=0)
mean_positive_embedding.shape, mean_negative_embedding.shape



for _ in range(5):
    pp(autoencoder.generate_from_latent(mean_positive_embedding))
for _ in range(5):
    pp(autoencoder.generate_from_latent(mean_negative_embedding))



start = 'Taylor Swift\'s artistic taste is characterized by her experimentation with different genres and her focus on pastel colors, vintage fashion, and whimsical imagery. Her attention to detail and unique visual aesthetic has helped her to stand out in the music industry.'
start_embedding = autoencoder.embed(start)

positive_to_negative = mean_negative_embedding - mean_positive_embedding

for t in torch.linspace(0, 2, 8):
    embedding = slerp(start_embedding, start_embedding + positive_to_negative, t)
    print(f'negative × {t:.2f}')
    pp(autoencoder.generate_from_latent(embedding))