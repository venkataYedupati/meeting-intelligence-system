from src.preprocess import (
    load_transcript,
    clean_text,
    save_cleaned_transcript,
)
from src.agentic import run_agentic_analysis
from src.output_formatter import save_meeting_output


def main() -> None:
    input_file = "data/raw/meeting1.txt"
    cleaned_output_file = "data/processed/meeting1_cleaned.txt"
    final_output_file = "data/processed/meeting1_full_output.json"

    transcript = load_transcript(input_file)
    cleaned_transcript = clean_text(transcript)
    save_cleaned_transcript(cleaned_transcript, cleaned_output_file)

    query = "What was decided about the demo?"
    final_output = run_agentic_analysis(transcript, query)

    save_meeting_output(final_output, final_output_file)

    print("\n" + "=" * 60)
    print("FINAL MEETING OUTPUT:\n")
    print(final_output)

    print("\n" + "=" * 60)
    print(f"Full output saved to: {final_output_file}")


if __name__ == "__main__":
    main()
