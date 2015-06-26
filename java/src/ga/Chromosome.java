package ga;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

public class Chromosome implements Comparable<Chromosome> {
	public static double MUTATION_RATE = 0.01;

	private static final Random random = new Random();

	private Set<Word> words; // genes
	private double fitness;

	public Chromosome() {
		int wordCount = random.nextInt(Word.ALL.size() - 2) + 2;
		words = new HashSet<Word>();
		for (int i = 0; i < wordCount; i++) {
			int index = random.nextInt(Word.ALL.size());
			words.add(Word.ALL.get(index));
		}
		while (words.size() < 2) {
			int index = random.nextInt(Word.ALL.size());
			words.add(Word.ALL.get(index));
		}
		computeFitness();
	}

	public Chromosome(Set<Word> words) {
		this.words = words;
		computeFitness();
	}

	public Set<Word> getWords() {
		return words;
	}

	private void computeFitness() {
		fitness = 1 + ((double) Math.max(0, words.size() - 2) * Math.max(0, getWriters().size() - 2) * getChars().size())
				/ (Word.ALL.size() * Word.ALL_WRITERS.size() * 72);
	}

	public Set<String> getWriters() {
		Set<String> commonWriters = null;
		for (Word word : words) {
			if (commonWriters == null)
				commonWriters = new HashSet<String>(word.getWriters());
			else
				commonWriters.retainAll(word.getWriters());
		}
		return commonWriters;
	}

	public Set<Character> getChars() {
		Set<Character> chars = new HashSet<Character>();
		for (Word word : words) {
			chars.addAll(word.getChars());
		}
		return chars;
	}

	public double getFitness() {
		return fitness;
	}

	public List<Chromosome> crossoverWith(Chromosome other) {
		if (words.size() < 2 || other.words.size() < 2)
			System.out.println("!!!!");
		int thisCutPos = random.nextInt(words.size() - 1) + 1;
		int otherCutPos = random.nextInt(other.words.size() - 1) + 1;

		Set<Word> words1 = new HashSet<Word>();
		Set<Word> words2 = new HashSet<Word>();
		int count = 0;
		for (Word word : words) {
			if (count <= thisCutPos)
				words1.add(word);
			else
				words2.add(word);
			count++;
		}
		count = 0;
		for (Word word : other.words) {
			if (count <= otherCutPos)
				words2.add(word);
			else
				words1.add(word);
			count++;
		}

		if (words1.size() < 2 || words2.size() < 2)
			System.out.println("????");

		return Arrays.asList(new Chromosome[] { new Chromosome(words1), new Chromosome(words2) });
	}

	public Chromosome mutate() {
		Set<Word> newWords = new HashSet<Word>();
		for (Word word : words) {
			if (random.nextDouble() < MUTATION_RATE) {
				int index = random.nextInt(Word.ALL.size());
				newWords.add(Word.ALL.get(index));
			} else
				newWords.add(word);
		}
		while (newWords.size() < 2) {
			int index = random.nextInt(Word.ALL.size());
			newWords.add(Word.ALL.get(index));
		}
		return new Chromosome(newWords);
	}

	@Override
	public int compareTo(Chromosome other) {
		return (int) Math.signum(other.getFitness() - this.getFitness());
	}

	@Override
	public String toString() {
		return words.toString();
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		result = prime * result + ((words == null) ? 0 : words.hashCode());
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Chromosome other = (Chromosome) obj;
		if (words == null) {
			if (other.words != null)
				return false;
		} else if (!words.equals(other.words))
			return false;
		return true;
	}
}