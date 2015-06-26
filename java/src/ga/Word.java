package ga;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

public class Word {
	public static ArrayList<Word> ALL = new ArrayList<Word>();
	public static Set<String> ALL_WRITERS = new HashSet<String>();

	private String text;
	private Set<Character> chars;
	private Set<String> writers;

	public Word(String text, Set<String> writers) {
		this.text = text;
		chars = new HashSet<Character>();
		for (int i = 0; i < text.length(); i++)
			chars.add(text.charAt(i));
		this.writers = writers;
		ALL_WRITERS.addAll(writers);
	}
	
	@Override
	public String toString() {
		return text;
	}
	
	public String getText() {
		return text;
	}
	
	public Set<Character> getChars() {
		return chars;
	}

	public Set<String> getWriters() {
		return writers;
	}
}
