"""
CLI utilities for managing the question bank.
"""

import argparse
from pathlib import Path
from .question_bank import QuestionBank
import json


def show_statistics(bank: QuestionBank) -> None:
    """Display question bank statistics."""
    stats = bank.get_statistics()
    
    print("\n📊 Question Bank Statistics")
    print("=" * 50)
    print(f"📍 Location: {stats['bank_location']}")
    print(f"📦 Total questions: {stats['total_questions']}")
    print(f"🆕 Never used: {stats['never_used']}")
    print(f"📅 Created: {stats['created_at']}")
    print(f"🔄 Last updated: {stats['last_updated']}")
    
    if stats['traits_covered']:
        print(f"\n🏷️ Traits covered: {len(stats['traits_covered'])}")
        for trait, count in stats['questions_per_trait'].items():
            print(f"   • {trait}: {count} questions used")
    else:
        print("\n🏷️ No traits covered yet")


def list_questions(bank: QuestionBank, limit: int = 10) -> None:
    """List questions from the bank."""
    questions = bank.bank_data["questions"][:limit]
    
    print(f"\n📝 Showing {min(limit, len(questions))} of {len(bank.bank_data['questions'])} questions:")
    print("=" * 50)
    
    for i, question in enumerate(questions, 1):
        usage = bank.bank_data["usage_tracking"].get(question, {})
        used_count = usage.get("usage_count", 0)
        traits = usage.get("used_for_traits", [])
        
        print(f"\n{i}. {question}")
        if used_count > 0:
            print(f"   Used {used_count} times for: {', '.join(traits)}")
        else:
            print("   Never used")


def export_bank(bank: QuestionBank, output_path: str) -> None:
    """Export the question bank to a file."""
    output_file = Path(output_path)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(bank.bank_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Exported question bank to {output_file}")


def import_questions(bank: QuestionBank, input_path: str) -> None:
    """Import questions from a file."""
    input_file = Path(input_path)
    
    if not input_file.exists():
        print(f"❌ File not found: {input_file}")
        return
    
    with open(input_file, 'r', encoding='utf-8') as f:
        if input_file.suffix == '.json':
            data = json.load(f)
            if isinstance(data, list):
                questions = data
            elif isinstance(data, dict) and 'questions' in data:
                questions = data['questions']
            else:
                print("❌ Invalid JSON format. Expected list of questions or object with 'questions' key.")
                return
        else:
            # Plain text, one question per line
            questions = [line.strip() for line in f if line.strip()]
    
    added = bank.add_questions(questions)
    print(f"✅ Imported {added} new questions (skipped {len(questions) - added} duplicates)")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Manage the Wisent Guard question bank")
    parser.add_argument(
        "--bank-path", 
        type=str, 
        help="Path to question bank JSON file (default: ~/.wisent-guard/question_bank.json)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Stats command
    subparsers.add_parser("stats", help="Show question bank statistics")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List questions from the bank")
    list_parser.add_argument("--limit", type=int, default=10, help="Number of questions to show")
    
    # Clear command
    subparsers.add_parser("clear", help="Clear all questions from the bank")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export question bank to file")
    export_parser.add_argument("output", help="Output file path")
    
    # Import command
    import_parser = subparsers.add_parser("import", help="Import questions from file")
    import_parser.add_argument("input", help="Input file path (JSON or plain text)")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize question bank
    bank = QuestionBank(args.bank_path)
    
    # Execute command
    if args.command == "stats":
        show_statistics(bank)
    elif args.command == "list":
        list_questions(bank, args.limit)
    elif args.command == "clear":
        response = input("⚠️ Are you sure you want to clear the question bank? (y/N): ")
        if response.lower() == 'y':
            bank.clear_bank()
        else:
            print("Cancelled.")
    elif args.command == "export":
        export_bank(bank, args.output)
    elif args.command == "import":
        import_questions(bank, args.input)


if __name__ == "__main__":
    main()