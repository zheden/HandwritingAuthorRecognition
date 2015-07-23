import java.io.File;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.List;

public class SelectWriterImages {

	public static void main(String[] args) throws Exception {
		new SelectWriterImages().gatherImages();
	}

	public void gatherImages() throws Exception {
		File dataDir = new File("/Users/GiK/Documents/TUM/Semester 4/MLfAiCV/Project/new-data2");

		File newDataDir = new File("/Users/GiK/Documents/TUM/Semester 4/MLfAiCV/Project/all-writers2");
//		String writersString = "348, 349"; //"332, 333, 334, 336, 337, 338, 339, 340, 341, 342, 343, 344, 346, 347";
//		List<String> writers = Arrays.asList(writersString.split(", "));

		File[] wordDirs = dataDir.listFiles();
		for (File wordDir : wordDirs) {
			if (wordDir.getName().startsWith("."))
				continue;
			File[] writerDirs = wordDir.listFiles();
			for (File writerDir : writerDirs) {
				if (writerDir.getName().startsWith("."))
					continue;
//				if (writers.contains(writerDir.getName())) {
					File newWriterDir = new File(newDataDir, writerDir.getName());
					File[] images = writerDir.listFiles();
					for (File image : images) {
						copyFile(image, newWriterDir);
					}
//				}
			}
		}
	
		System.out.println("Done!");
	}

	private void copyFile(File file, File destDir) throws Exception {
		destDir.mkdirs();
		Files.copy(file.toPath(), destDir.toPath().resolve(file.getName()));
	}
}
