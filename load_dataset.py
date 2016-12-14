import random
import numpy as np
import pickle

all_recipes = pickle.load(open('dataset/cleaned_recipes.p', 'rb'))
char2id = pickle.load(open('dataset/char2id.p', 'rb'))
id2char = pickle.load(open('dataset/id2char.p', 'rb'))
max_recipe_length = 500
num_recipes = 30724

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
        
def booze_permuter(recipes):
    while True:
        for rec in recipes:
            ings = rec['ingredients']
            i = 0
            while i < len(ings) and is_number(ings[i][0]):
                i += 1

            ing_list = ings[:i]
            garn_list = ings[i:]

            random.shuffle(ing_list)
            random.shuffle(garn_list)

            ing_list.extend(garn_list)
            yield '\n'.join(ing_list)
        
def training_set_generator(num_recipes):
    recipe_generator = booze_permuter(all_recipes)
    while True:
        recipe_list = []
        while len(recipe_list) < num_recipes:
            recipe_list.append(next(recipe_generator))
            
        maxSequenceLength = max_recipe_length + 1
        inputChars = np.zeros((num_recipes, maxSequenceLength, len(char2id)), dtype=np.bool)
        nextChars = np.zeros((num_recipes, maxSequenceLength, len(char2id)), dtype=np.bool)

        for i in range(0, num_recipes):
            inputChars[i, 0, char2id['S']] = 1
            nextChars[i, 0, char2id[recipe_list[i][0]]] = 1
            for j in range(1, maxSequenceLength):
                if j < len(recipe_list[i]) + 1:
                    inputChars[i, j, char2id[recipe_list[i][j - 1]]] = 1
                    if j < len(recipe_list[i]):
                        nextChars[i, j, char2id[recipe_list[i][j]]] = 1
                    else:
                        nextChars[i, j, char2id['E']] = 1
                else:
                    inputChars[i, j, char2id['E']] = 1
                    nextChars[i, j, char2id['E']] = 1
        
        yield (inputChars, nextChars)
