import java.io.File;
import java.nio.file.Files;
import java.util.HashSet;
import java.util.Set;

public class SelectWordImages {

	public static void main(String[] args) throws Exception {
		new SelectWordImages().gatherImages();
	}

	public void gatherImages() throws Exception {
		File dataDir = new File("/Users/GiK/Documents/TUM/Semester 4/MLfAiCV/Project/new-data-clean");

		File newDataDir = new File("/Users/GiK/Documents/TUM/Semester 4/MLfAiCV/Project/new-data-set1");
		String wordsString = "be, disciples, as, damp, Commons, is, health, usual, thought, Hahnemann, knows, cultivated,"
				+ " writing, hoped, Sir, this, Arthur, season, year, wondered, human, remarkable, agree, distinguished,"
				+ " There, Italian, sent, upon, which, Anglesey, country, Leipzig, number, give, success, have, town,"
				+ " for, nor, early, Samuel, Whigs, truly, he, that, Lord, by, happens, back, than, doctor, These";
		String[] words = wordsString.split(", ");

		Set<String> writers = null;
		for (int i = 0; i < words.length; i++) {
			String word = words[i];
			File wordDir = new File(dataDir, word);
			File[] writerDirs = wordDir.listFiles();
			Set<String> wordWriters = new HashSet<String>();
			for (int j = 0; j < writerDirs.length; j++) {
				if (word.startsWith("."))
					continue;
				wordWriters.add(writerDirs[j].getName());
			}
			System.out.println("Found word '" + word + "' with writers " + wordWriters);
			if (writers == null)
				writers = wordWriters;
			else
				writers.retainAll(wordWriters);
		}
		
		System.out.println("Writers are " + writers);
		for (int i = 0; i < words.length; i++) {
			String word = words[i];
			for (String writer : writers) {
				File writerDir = new File(dataDir, word + "/" + writer);
				File newWriterDir = new File(newDataDir, word + "/" + writer);
				File[] images = writerDir.listFiles();
				for (int j = 0; j < images.length; j++) {
					copyFile(images[j], newWriterDir);
				}
			}
		}
		
		System.out.println("Done!");
	}

	private void copyFile(File file, File destDir) throws Exception {
		destDir.mkdirs();
		Files.copy(file.toPath(), destDir.toPath().resolve(file.getName()));
	}
}
