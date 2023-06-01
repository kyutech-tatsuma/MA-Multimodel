import os
import csv
import argparse

def makecsv(opt):
    # プログラムの置かれているディレクトリのパスを取得
    program_directory = os.path.dirname(os.path.abspath(__file__))
    # フォルダのパスを指定
    folder_path = opt.folder

    # ファイル名を取得
    file_names = os.listdir(folder_path)

    # CSVファイルに相対パスを書き込む
    csv_file_path = opt.result

    with open(csv_file_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        for file_name in file_names:
            # ファイルの相対パスを取得
            file_path = os.path.join(folder_path, file_name)
            relative_path = os.path.relpath(file_path, program_directory)
            
            # CSVファイルに相対パスを書き込む
            writer.writerow([relative_path])

    print("CSVファイルの作成が完了しました。")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True, help='target folder')
    parser.add_argument("--result", type=str, required=True, help="result path")
    opt = parser.parse_args()
    print(opt)
    makecsv(opt)
