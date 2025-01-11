import csv
import re

def convert_txt_to_csv(input_file_path, output_file_path):
    # Read the input file
    try:
        with open(input_file_path, 'r', encoding='utf-8') as file:
            input_text = file.read()
        
        # Regex pattern to extract questions and answers
        pattern = r'<question>(.*?)<answer>(.*?)<end>'
        
        # Find all matches
        qa_pairs = re.findall(pattern, input_text, re.DOTALL)
        
        # Write to CSV
        with open(output_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(['question', 'answer'])
            # Write data
            for question, answer in qa_pairs:
                writer.writerow([question.strip(), answer.strip()])
                
        print(f"İşlem tamamlandı. {len(qa_pairs)} adet soru-cevap çifti CSV dosyasına yazıldı.")
        print(f"CSV dosyası oluşturuldu: {output_file_path}")
                
    except FileNotFoundError:
        print("Girdi dosyası bulunamadı!")
    except Exception as e:
        print(f"Bir hata oluştu: {str(e)}")

# Kullanım örneği
input_file = "dataset/finished-dataset.txt"  # txt dosyanızın adı
output_file = "dataset/finished-qa_dataset.csv"  # oluşturulacak csv dosyasının adı

convert_txt_to_csv(input_file, output_file)