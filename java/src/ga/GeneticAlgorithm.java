package ga;


public class GeneticAlgorithm {
	public static int POPULATION_SIZE = 200;
	public static int MAX_GENERATIONS = 5000;
	
	private Population population;
	private int generation;

	public GeneticAlgorithm() {
		population = new Population(POPULATION_SIZE);
	}

	public Population train() {
		population = new Population(POPULATION_SIZE);

		generation = 0;
		while (generation < MAX_GENERATIONS) {
			if (generation % 100 == 0)
				System.out.println("Best fitness: " + population.getBestFitness() + ", average fitness: " + population.getAverageFitness());
			population = population.getNextGeneration();
			generation++;
		}

		return population;
	}
}
