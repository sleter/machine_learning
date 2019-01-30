import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# n - skończona populacja
# gra jest 2 na 2 więcej initial musi być macierzą 2-2
# i - ile jest populacji typu pierwszego (w tym przypadku zer)

def moran_process(n, i, initial):
    population = [0 for _ in range(i)] + [1 for _ in range(n-i)]
    # zliczamy ile jest zer, a ile jedynek
    counts = []
    count_append(population, counts)

    # tworzone są kolejne populacje do momentu gdzie będą w populacji same zera lub jedynki
    while all(x in population for x in [0, 1]):
        scores = every_player_vs_every_player(population, initial)

        total_score = sum(scores)
        # prawdopodobieństwa otrzymania danych wyników
        probabilities = [score / total_score for score in scores]
        # wybierany jest index elementu który będzie repordukowany biorąc pod uwagę prawdopodobieństwa
        birth_index = np.random.choice(range(n), p=probabilities)

        # losowy indeks do usunięcia
        death_index = np.random.randint(n)

        # wartość wynikająca z reprodukcji(birth) zastępuje tą która jest wyeliminowana(death)
        population[death_index] = population[birth_index]

        count_append(population, counts)
    return counts

def count_append(population, counts):
    counts.append((population.count(0), population.count(1)))

def every_player_vs_every_player(population, initial):
    scores = []
    for i, player1 in  enumerate(population):
        total = 0
        for j, player2 in enumerate(population):
            if i != j:
                total += initial[player1, player2]
        scores.append(total)
    return scores

counts = moran_process(7, 3, np.array([[2,1], [1,2]]))
plt.plot(counts)
plt.show()
