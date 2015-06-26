package ga;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class Population {
	private static final double ELITE_PERCENTAGE = 0.1;
	private static final double NEWCOMERS_PERCENTAGE = 0.1;

	private static final Random random = new Random();

	private List<Chromosome> chromosomes;

	public Population(int size) {
		chromosomes = new ArrayList<>(size);
		for (int i = 0; i < size; i++)
			chromosomes.add(new Chromosome());
	}

	public Population(List<Chromosome> population) {
		this.chromosomes = population;
		Collections.sort(population);
	}

	public Population getNextGeneration() {
		double totalFitness = 0;
		for (Chromosome chromosome : chromosomes)
			totalFitness += chromosome.getFitness();
		LinkedHashMap<Chromosome, Double> roulette = new LinkedHashMap<Chromosome, Double>();
		double currentFitness = 0;
		for (Chromosome chromosome : chromosomes) {
			currentFitness += chromosome.getFitness();
			roulette.put(chromosome, currentFitness / totalFitness);
		}
		
		List<Chromosome> newPopulation = new ArrayList<>(chromosomes.size());
		newPopulation.addAll(chromosomes.subList(0, (int) (chromosomes.size() * ELITE_PERCENTAGE)));
		for (int i = 0; i < (int) chromosomes.size() * NEWCOMERS_PERCENTAGE; i++) {
			Chromosome chromosome = new Chromosome();
			if (!newPopulation.contains(chromosome))
				newPopulation.add(chromosome);
		}

		while (newPopulation.size() < chromosomes.size()) {
			// crossover
			Chromosome parent1 = selectParent(roulette);
			Chromosome parent2 = selectParent(roulette);
			List<Chromosome> offspring = parent1.crossoverWith(parent2);

			// mutation
			for (Chromosome child : offspring) {
				Chromosome chromosome = child.mutate();
				if (!newPopulation.contains(chromosome) && newPopulation.size() < chromosomes.size())
					newPopulation.add(chromosome);
			}
		}

		return new Population(newPopulation);
	}
	
	private Chromosome selectParent(LinkedHashMap<Chromosome, Double> roulette) {
		double sample = random.nextDouble();
		for (Map.Entry<Chromosome, Double> entry : roulette.entrySet()) {
			if (sample < entry.getValue())
				return entry.getKey();
		}
		return roulette.keySet().iterator().next();
	}

	public double getAverageFitness() {
		double fitness = 0;
		for (Chromosome chromosome : chromosomes)
			fitness += chromosome.getFitness();
		return fitness / chromosomes.size();
	}

	public double getBestFitness() {
		return getBestChromosome().getFitness();
	}

	public Chromosome getBestChromosome() {
		return chromosomes.get(0);
	}
	
	public List<Chromosome> getChromosomes() {
		return chromosomes;
	}
}
