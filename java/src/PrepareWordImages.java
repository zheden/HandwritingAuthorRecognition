import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.nio.file.Files;
import java.util.HashMap;
import java.util.Map;

public class PrepareWordImages {

	public static void main(String[] args) throws Exception {
		new PrepareWordImages().gatherImages();
	}

	public void gatherImages() throws Exception {
		File dataDir = new File("/Users/GiK/Documents/TUM/Semester 4/MLfAiCV/Project/data");
		File asciiDir = new File(dataDir, "ascii");
		File wordsDir = new File(dataDir, "words");

		File newDataDir = new File("/Users/GiK/Documents/TUM/Semester 4/MLfAiCV/Project/new-data2");

		Map<String, Integer> writerIds = readFormToWriterId(asciiDir);
//		Map<String, Integer> wordCount = readWordCount(asciiDir);
		
//		int atLeastCount = 1;
		
		BufferedReader reader = new BufferedReader(new FileReader(new File(asciiDir, "words.txt")));
		String line;
		while ((line = reader.readLine()) != null) {
			if (line.startsWith("#"))
				continue;
			
			String[] tokens = line.split(" ");
			String word = tokens[tokens.length - 1];
			if (!Character.isLetter(word.charAt(0)))
				continue;
//			char ch = word.charAt(0);
//			if (wordCount.get(word) < atLeastCount || !Character.isLetter(ch))
//				continue;
			
			String wordId = tokens[0];
			String formId = wordId.substring(0, wordId.length() - 6);
			String prefix = formId.substring(0, formId.indexOf('-'));
			File wordFile = new File(wordsDir, prefix + '/' + formId + '/' + wordId + ".png");
			
			Integer writerId = writerIds.get(formId);
			File destDir = new File(newDataDir, word + '/' + writerId);
			copyFile(wordFile, destDir);
		}
		reader.close();
		
		System.out.println("Done!");
	}

	private void copyFile(File file, File destDir) throws Exception {
		destDir.mkdirs();
		Files.copy(file.toPath(), destDir.toPath().resolve(file.getName()));
	}

	private Map<String, Integer> readFormToWriterId(File asciiDir) throws Exception {
		Map<String, Integer> writerIds = new HashMap<String, Integer>();

		BufferedReader reader = new BufferedReader(new FileReader(new File(asciiDir, "forms.txt")));
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

		return writerIds;
	}

	private Map<String, Integer> readWordCount(File asciiDir) throws Exception {
		Map<String, Integer> wordCount = new HashMap<String, Integer>();

		BufferedReader reader = new BufferedReader(new FileReader(new File(asciiDir, "words.txt")));
		String line;
		while ((line = reader.readLine()) != null) {
			if (line.startsWith("#"))
				continue;

			String[] tokens = line.split(" ");
			String word = tokens[tokens.length - 1];

			if (wordCount.containsKey(word))
				wordCount.put(word, wordCount.get(word) + 1);
			else
				wordCount.put(word, 1);
		}
		reader.close();
		
		return wordCount;
	}
}
