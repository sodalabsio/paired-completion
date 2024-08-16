import yaml


def import_climate():
    CORPUS = "/workspaces/dev/projects/narratives/classifiers/real_world_corpora/with_jensen_garrett_abbott_climate_climate_change_pms_curie_2_-1.json"

    with open(CORPUS, "r") as f:
        corpus_data = yaml.safe_load(f)

    # Gather texts
    X = [text["text"] for text in corpus_data]
    y = [text["speaker"] for text in corpus_data]
    y = ["a" for _ in y]

    # Gather seeds
    # seeds = [
    #     {
    #         "a": "An overwhelming majority of climate scientists, over 97%, agree that climate change is real and primarily caused by human activities, particularly the emission of greenhouse gases.",
    #         "b": "Climate change can be seen as a part of Earth's natural cycle, and the extent to which human activities influence this process is still a subject of debate within the scientific community.",
    #     },
    #     {
    #         "a": "The current rapid increase in global temperatures is unprecedented and largely driven by human actions, distinguishing it from natural climate variability.",
    #         "b": "Despite the consensus narrative, some argue that climate change discussions are ongoing and that there is not complete agreement on the human impact.",
    #     },
    #     {
    #         "a": "Excessive concentrations of carbon dioxide in the atmosphere, while necessary for plant life, act as a potent greenhouse gas that traps heat and contributes to global warming.",
    #         "b": "The occurrence of cold weather events and chilly temperatures can be used to question the assertion that there is a gradual increase in global temperatures due to human actions.",
    #     },
    #     {
    #         "a": "The impacts of climate change are global and can affect every individual through extreme weather events, health risks, and economic changes, regardless of personal experience.",
    #         "b": "The ability of renewable energy sources to fully replace fossil fuels is contested, with skeptics questioning their efficiency and overall potential to meet global energy demands.",
    #     },
    #     {
    #         "a": "Renewable energy technologies are advancing and becoming more efficient, offering a promising alternative to fossil fuels and a pathway to mitigating climate change.",
    #         "b": "Some believe that organisms have the capacity to adapt to changes in their climate, suggesting that the effects of climate change may not be as dire as often portrayed.",
    #     },
    # ]

    seeds = [
        {
            "a": "Climate change is primarily caused by human activities, such as burning fossil fuels and deforestation.",
            "b": "Climate change is a natural phenomenon that has been happening for millions of years, and human impact is negligible.",
        },
        {
            "a": "Immediate action is required to combat climate change, including transitioning to renewable energy sources.",
            "b": "The economic costs of transitioning to renewable energy are too high, and such drastic measures are unnecessary.",
        },
        {
            "a": "Climate change poses a significant threat to global biodiversity, leading to the extinction of many species.",
            "b": "Species have always adapted to climate changes over the millennia; current changes are no different.",
        },
        {
            "a": "Investing in green technology is essential for combating climate change and can also drive economic growth.",
            "b": "Investments in green technology are risky and divert funds from more immediate economic needs.",
        },
        {
            "a": "International cooperation is crucial for effective action against climate change.",
            "b": "Each country should be free to pursue its own economic interests without being constrained by international agreements on climate change.",
        },
    ]

    names = {"a": "science", "b": "skepticism"}

    distilled = {
        "a": [
            # "I am a climate scientist.",
            # "I am a climate change activist.",
        ],
        "b": [
            # "I am a climate change skeptic.",
            # "I am a climate change denier.",
        ],
    }

    summarized = {"a": [], "b": []}

    return X, y, seeds, distilled, summarized, names


def import_asylum_seekers():
    CORPUS = "/workspaces/dev/projects/narratives/classifiers/real_world_corpora/Asylum_Seekers_withGreens_filtered_chars150to1200_reduced.json"

    with open(CORPUS, "r") as f:
        corpus_data = yaml.safe_load(f)

    # Gather texts
    X = [text["text"] for text in corpus_data]
    y = [text["speakername"] for text in corpus_data]
    y = ["a" for _ in y]

    # Gather seeds
    seeds = [
        {
            "a": "We must remember that asylum seekers are escaping dire situations where their basic human rights are often violated; it's our duty to offer them sanctuary.",
            "b": "It is crucial to maintain stringent border controls to prevent the entry of potential threats to our national security, ensuring the safety of all Australians.",
        },
        {
            "a": "As a signatory to the UN Refugee Convention, Australia has a legal and ethical obligation to protect those who arrive on our shores seeking refuge from persecution.",
            "b": "The economic burden of supporting a large number of asylum seekers could strain our public services and welfare systems, impacting the financial stability of our country.",
        },
        {
            "a": "It is our moral obligation as a prosperous and stable nation to extend a helping hand to those in need, showing compassion beyond our borders.",
            "b": "We need to uphold the integrity of our legal immigration system; allowing people to bypass it by arriving irregularly sets a precedent that could undermine the entire framework.",
        },
        {
            "a": "Asylum seekers bring a wealth of diverse cultural perspectives and skills that can greatly benefit our society and economy, particularly in sectors with labor shortages.",
            "b": "Australia must have the sovereign right to control its borders and decide who can enter and stay in our country, to preserve the order and effectiveness of our immigration policies.",
        },
        {
            "a": "We must ensure our asylum processing system is fair and transparent, adhering to international laws and respecting the dignity of each individual.",
            "b": "Implementing strict measures against irregular maritime arrivals is essential to dismantle human smuggling networks and protect vulnerable individuals from exploitation.",
        },
    ]

    distilled = {
        "a": [],
        "b": [],
    }

    summarized = {
        "a": [],
        "b": [],
    }

    names = {"a": "Humanitarian", "b": "Security"}

    return X, y, seeds, distilled, summarized, names


def import_data(CORPUS):
    if CORPUS in ("voice", "climate", "asylum_seekers"):
        if CORPUS == "voice":
            CORPUS = "/workspaces/dev/projects/narratives/classifiers/real_world_corpora/the_voice_the_voice_broad_keyscheck_4sep2023_filtered_chars150to1200_gpt-3.5-turbo-instruct_2_2 (1).json"
        elif CORPUS == "climate":
            return import_climate()
        elif CORPUS == "asylum_seekers":
            return import_asylum_seekers()
        else:
            raise ValueError("Invalid corpus: {}".format(CORPUS))

        with open(CORPUS, "r") as f:
            corpus_data = yaml.safe_load(f)

            # print(corpus_data)
            X = [text["text"] for text in corpus_data]
            y = [
                (text["affiliation"] if "affiliation" in text else text["speaker"])
                for text in corpus_data
            ]

            # # Counter({'Australian Labor Party': 561, 'Liberal Party of Australia': 95, 'Independent': 89, 'Australian Greens': 14, 'National Party of Australia': 10})
            # left = ["Australian Labor Party", "Australian Greens"]
            # right = ["Liberal Party of Australia", "National Party of Australia"]

            # # Merge into "left" and "right" affiliations, dropping others
            # y = [
            #     (
            #         "left"
            #         if affiliation in left
            #         else "right" if affiliation in right else None
            #     )
            #     for affiliation in y
            # ]
            # X = [x for x, y in zip(X, y) if y is not None]
            # y = [y for y in y if y is not None]
            # print(len(X), len(y))

            # # Drop any classes with less than 2 examples
            # from collections import Counter

            # counter = Counter(y)
            # print(counter)
            # X = [x for x, y in zip(X, y) if counter[y] > 1]
            # y = [y for y in y if counter[y] > 1]
            # print(len(X), len(y))

            # # Map left = a, right = b
            # y_s = []
            # for affiliation in y:
            #     if affiliation == "left":
            #         y_s.append("a")
            #     elif affiliation == "right":
            #         y_s.append("b")
            #     else:
            #         raise ValueError("Invalid affiliation: {}".format(affiliation))

            # y = y_s

            y = ["a"] * len(y)

            names = {"a": "for", "b": "against"}
            # seeds = {"a": [
            #     "Voting \"Yes\" is about acknowledging Aboriginal and Torres Strait Islander individuals in the Constitution and showing respect for their culture and traditions.",
            #     "The Voice that is being proposed will consist of a committee comprised of Aboriginal and Torres Strait Islander individuals. Their role will be to provide advice to the Parliament and Government regarding matters that impact their community.",
            #     "The Voice seeks to tackle issues that Aboriginal and Torres Strait Islander individuals encounter, including shorter life expectancy, increased disease rates, and limited educational prospects.",
            #     "The Voice will offer guidance to the government, resulting in improved decision-making, outcomes, and cost-effectiveness.",
            #     "The concept of the Voice originates directly from Aboriginal and Torres Strait Islander individuals and is backed by a majority of them.",
            #     "Constitutional recognition is considered a strong declaration that will facilitate tangible transformation and commemorate the rich heritage of Aboriginal and Torres Strait Islander communities.",
            #     "The Voice is anticipated to bring about tangible enhancements in life expectancy, health, education, and employment for Aboriginal and Torres Strait Islander individuals.",
            #     "Voting \"Yes\" is considered an act of unity and reconciliation that will foster togetherness among Australians.",
            #     "The Voice is anticipated to result in increased utilization of funding and improved outcomes.",
            #     "The Voice is considered a lasting solution that will offer stability and independence, providing practical advice without being entangled in immediate political matters.",
            # ], "b": [
            #     "The suggested Voice carries legal risks and may result in a prolonged period of litigation involving constitutional and administrative law.",
            #     "The operational details and member selection process of the Voice are not disclosed prior to the vote, which leaves Australians voting for an unfamiliar entity.",
            #     "The proposal is controversial because it establishes a body for a specific group of Australians in the Constitution, resulting in varying levels of citizenship.",
            #     "The Voice is irrevocable once enshrined in the Constitution, potentially resulting in irreversible negative outcomes.",
            #     "The Voice encompasses all aspects of \"Executive Government\", indicating that no matter the issue, it has the potential to result in legal disputes, delays, and an inefficient government.",
            #     "The Voice has the potential to encourage activists to advocate for additional transformative measures, such as reparations, compensation, and a reevaluation of Australian history.",
            #     "The Voice could potentially add another layer of bureaucracy, which may be expensive and inefficient, without replacing any current Indigenous representative organizations.",
            #     "The Voice is a permanent program in the Constitution, making it irreversible and binding Australia to its long-term repercussions.",
            #     "The referendum is not only about acknowledging Indigenous Australians in the Constitution, which could be done without a risky, unknown, and permanent Voice.",
            #     "The process leading up to the referendum was expedited and assertive, disregarding the valid inquiries and worries of numerous Australians.",
            # ]}
            seeds = [
                {
                    "a": 'Voting "Yes" is about acknowledging Aboriginal and Torres Strait Islander individuals in the Constitution and showing respect for their culture and traditions.',
                    "b": "The suggested Voice carries legal risks and may result in a prolonged period of litigation involving constitutional and administrative law.",
                },
                {
                    "a": "The Voice that is being proposed will consist of a committee comprised of Aboriginal and Torres Strait Islander individuals. Their role will be to provide advice to the Parliament and Government regarding matters that impact their community.",
                    "b": "The operational details and member selection process of the Voice are not disclosed prior to the vote, which leaves Australians voting for an unfamiliar entity.",
                },
                {
                    "a": "The Voice seeks to tackle issues that Aboriginal and Torres Strait Islander individuals encounter, including shorter life expectancy, increased disease rates, and limited educational prospects.",
                    "b": "The proposal is controversial because it establishes a body for a specific group of Australians in the Constitution, resulting in varying levels of citizenship.",
                },
                {
                    "a": "The Voice will offer guidance to the government, resulting in improved decision-making, outcomes, and cost-effectiveness.",
                    "b": "The Voice is irrevocable once enshrined in the Constitution, potentially resulting in irreversible negative outcomes.",
                },
                {
                    "a": "The concept of the Voice originates directly from Aboriginal and Torres Strait Islander individuals and is backed by a majority of them.",
                    "b": 'The Voice encompasses all aspects of "Executive Government", indicating that no matter the issue, it has the potential to result in legal disputes, delays, and an inefficient government.',
                },
                {
                    "a": "Constitutional recognition is considered a strong declaration that will facilitate tangible transformation and commemorate the rich heritage of Aboriginal and Torres Strait Islander communities.",
                    "b": "The Voice has the potential to encourage activists to advocate for additional transformative measures, such as reparations, compensation, and a reevaluation of Australian history.",
                },
                {
                    "a": "The Voice is anticipated to bring about tangible enhancements in life expectancy, health, education, and employment for Aboriginal and Torres Strait Islander individuals.",
                    "b": "The Voice could potentially add another layer of bureaucracy, which may be expensive and inefficient, without replacing any current Indigenous representative organizations.",
                },
                {
                    "a": 'Voting "Yes" is considered an act of unity and reconciliation that will foster togetherness among Australians.',
                    "b": "The Voice is a permanent program in the Constitution, making it irreversible and binding Australia to its long-term repercussions.",
                },
                {
                    "a": "The Voice is anticipated to result in increased utilization of funding and improved outcomes.",
                    "b": "The referendum is not only about acknowledging Indigenous Australians in the Constitution, which could be done without a risky, unknown, and permanent Voice.",
                },
                {
                    "a": "The Voice is considered a lasting solution that will offer stability and independence, providing practical advice without being entangled in immediate political matters.",
                    "b": "The process leading up to the referendum was expedited and assertive, disregarding the valid inquiries and worries of numerous Australians.",
                },
            ]
            distilled = {
                "a": [
                    # "I am a member of the Australian Labor Party.",
                    # "I am a member of the Australian Greens.",
                ],
                "b": [
                    # "I am a member of the Liberal Party of Australia.",
                    # "I am a member of the National Party of Australia.",
                ],
            }
            summarized = {"a": [], "b": []}
    else:

        with open(CORPUS, "r") as f:
            corpus_data = yaml.safe_load(f)

        seeds = corpus_data["seeds"]
        distilled = corpus_data["distilled"]
        summarized = corpus_data["summarized"]
        names = corpus_data["names"]
        dataset = corpus_data["dataset"]

        # Convert the dataset to corpus data format
        corpus_data = []
        for seed_set in dataset:
            for a in seed_set["a_first"]["a"]:
                corpus_data.append({"text": a, "speakername": "a"})
            for b in seed_set["a_first"]["b"]:
                corpus_data.append({"text": b, "speakername": "b"})
            for a in seed_set["b_first"]["a"]:
                corpus_data.append({"text": a, "speakername": "a"})
            for b in seed_set["b_first"]["b"]:
                corpus_data.append({"text": b, "speakername": "b"})

        # Shuffle
        # random.shuffle(corpus_data)

        print("Loaded {} texts from corpus".format(len(corpus_data)))
        print(
            "Total word count:",
            sum([len(text["text"].split()) for text in corpus_data]),
        )

        # Create the training data
        X = [text["text"] for text in corpus_data]
        y = [text["speakername"] for text in corpus_data]
        print(len(X), len(y))

    return X, y, seeds, distilled, summarized, names
