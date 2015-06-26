package ga;

import java.io.File;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class MaximiseWritersWords {

	public static void main(String[] args) {
		File rootDir = new File("/Users/GiK/Documents/TUM/Semester 4/MLfAiCV/Project/new-data-clean");
		File[] wordDirs = rootDir.listFiles();
		for (int i = 0; i < wordDirs.length; i++) {
			String word = wordDirs[i].getName();
			if (word.startsWith("."))
				continue;
			File[] writerDirs = wordDirs[i].listFiles();
			Set<String> writers = new HashSet<String>();
			for (int j = 0; j < writerDirs.length; j++) {
				if (writerDirs[j].getName().startsWith("."))
					continue;
				writers.add(writerDirs[j].getName());
			}
			System.out.println("Found word '" + word + "' with writers " + writers);
			Word.ALL.add(new Word(word, writers));
		}
		List<Chromosome> chromosomes = new GeneticAlgorithm().train().getChromosomes();
		for (int i = 0; i < 5; i++) {
			System.out.println(chromosomes.get(i).getWords() + " -> " + chromosomes.get(i).getWriters().size());
		}
	}

}
