Input Data

Recipe
- Ingredients
    - 1/2 cup butter
    - 1/2 cup white sugar
    - 1/2 cup packed brown sugar
    - 1/2 teaspoon vanilla extract
    - 1 egg
    - 1 teaspoon baking soda
    - 1/2 teaspoon hot water
    - 1/4 teaspoon salt
    - 1 1/2 cups all-purpose flour
    - 1 cup semisweet chocolate chips
- Directions
    - Preheat oven to 350 degrees F (175 degrees C).
    - Cream together the butter, white sugar, and brown sugar until smooth.
    - Beat in the eggs one at a time, then stir in the vanilla.
    - Dissolve baking soda in hot water.
    - Add to batter along with salt.
    - Stir in flour, chocolate chips, and nuts.
    - Drop by large spoonfuls onto ungreased pans.
    - Bake for about 10 minutes in the preheated oven, or until edges are nicely browned.
    - Cool on wire racks.

df = [
    {
        "date": "1893-01-01",
        "book": "...",
        "title": "Chocolate Chip Cookies I",
        "ingredients": [
            {
                "name": "butter",
                "quantity": "1/2 cup",
                "critical": true,
                "alternatives": [
                    {
                        "name": "margarine",
                        "quantity": "1/2 cup"
                    },
                    {
                        "name": "shortening",
                        "quantity": "1/2 cup"
                    }
                ]
            },
            ...
        ],
        "directions": [
            {
                "step": 1,
                "text": "Preheat oven to 350 degrees F (175 degrees C)."
            },
            ...
        ]
    }
]

Book = {
    A: "Preamble" (50 tokens),
    B: "Chapter 1, Page 1-2: Cultural Introduction" (40 tokens),
    C: "Recipe 1: Chocolate Chip Cookies I" (60 tokens),
    D: "Recipe 2: Chocolate Chip Cookies II" (20 tokens),
    ...
}

Process:
1. Embed every segment of the book
2. Create a complete graph with distances from each segment to each other segment

E.g.
A -> B: 0.7
A -> C: 0.15
A -> C: 0.2
B -> C: 0.35
B -> D: 0.6
C -> D: 0.5

Suppose we want to describe the recipe for chocolate chip cookies I, in chunk C.

Token Budget: 100

Chunk C = 60 tokens
Take chunk D = 60 + 20 tokens = 80 tokens
Take chunk B = 80 + 40 tokens = 120 tokens
Over the limit, so stop

*** Context ***
Chunk D

*** Recipe ***
Chunk C

Ask the model for whatever output you want.

---

For each book:
- Is there a recipe in this book?
  - For each chunk, does this chunk contain a whole recipe and/or part of a recipe?
- For each recipe, extract:
  - Ingredients + Quantities
  - Steps
  - Techniques used

Ideal outcome: Insight into how foods change across cultures and/or across time.