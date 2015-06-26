import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class OriginalDataAnalyzer {

	public static void main(String[] args) throws Exception {
		new OriginalDataAnalyzer().analyseWordsWriters(new File("/Users/GiK/Documents/TUM/Semester 4/MLfAiCV/Project/data/ascii"));
	}

	public void analyzeLines(File dataDir) throws Exception {
		Map<Integer, Integer> writerLines = new HashMap<Integer, Integer>();
		Map<String, Integer> forms = new HashMap<String, Integer>();

		BufferedReader reader = new BufferedReader(new FileReader(new File(dataDir, "forms.txt")));
		String line;
		while ((line = reader.readLine()) != null) {
			if (line.startsWith("#"))
				continue;

			String[] tokens = line.split(" ");
			String formId = tokens[0];
			int writerId = Integer.parseInt(tokens[1]);
			forms.put(formId, writerId);
			writerLines.put(writerId, 0);
		}
		reader.close();

		reader = new BufferedReader(new FileReader(new File(dataDir, "lines.txt")));
		while ((line = reader.readLine()) != null) {
			if (line.startsWith("#"))
				continue;

			String[] tokens = line.split(" ");
			String lineId = tokens[0];
			String formId = lineId.substring(0, lineId.lastIndexOf('-'));
			Integer writerId = forms.get(formId);
			writerLines.put(writerId, writerLines.get(writerId) + 1);
		}
		reader.close();

		System.out.println(writerLines.values());
	}

	public void analyzeWords(File dataDir) throws Exception {
		Map<String, Integer> words = new HashMap<String, Integer>();

		BufferedReader reader = new BufferedReader(new FileReader(new File(dataDir, "words.txt")));
		String line;
		while ((line = reader.readLine()) != null) {
			if (line.startsWith("#"))
				continue;

			String[] tokens = line.split(" ");
			String word = tokens[tokens.length - 1];
			if (words.containsKey(word))
				words.put(word, words.get(word) + 1);
			else
				words.put(word, 1);
		}
		reader.close();

		System.out.println(words.values());
	}

	public void analyseWordsWriters(File dataDir) throws Exception {
		Map<String, Integer> writerIds = new HashMap<String, Integer>();
		final Map<String, Set<Integer>> wordWriters = new HashMap<String, Set<Integer>>();
		Map<String, Integer> wordCount = new HashMap<String, Integer>();

		BufferedReader reader = new BufferedReader(new FileReader(new File(dataDir, "forms.txt")));
		String line;
		while ((line = reader.readLine()) != null) {
			if (line.startsWith("#"))
				continue;

			String[] tokens = line.split(" ");
			String formId = tokens[0];
			int writerId = Integer.parseInt(tokens[1]);
			writerIds.put(formId, writerId);
		}
		reader.close();

		reader = new BufferedReader(new FileReader(new File(dataDir, "words.txt")));
		while ((line = reader.readLine()) != null) {
			if (line.startsWith("#"))
				continue;

			String[] tokens = line.split(" ");
			String word = tokens[tokens.length - 1];
			String formId = tokens[0].substring(0, tokens[0].length() - 6);
			Integer writerId = writerIds.get(formId);
			if (writerId == null)
				throw new Exception("writer not found for form id " + formId);

			if (!wordWriters.containsKey(word))
				wordWriters.put(word, new HashSet<Integer>());
			Set<Integer> writers = wordWriters.get(word);
			writers.add(writerId);
			
			if (wordCount.containsKey(word))
				wordCount.put(word, wordCount.get(word) + 1);
			else
				wordCount.put(word, 1);
		}
		reader.close();

		List<String> words = new ArrayList<String>(wordWriters.keySet());
		Collections.sort(words, new Comparator<String>() {
			@Override
			public int compare(String o1, String o2) {
				return wordCount.get(o2) - wordCount.get(o1);
			}
		});

		for (int i = 0; i < 50; i++) {
			String word = words.get(i);
			//if (word.length() > 4)
				System.out.println(word + " -  " + wordCount.get(word) + " images by " + wordWriters.get(word).size() + " writers");
		}
	}
}
