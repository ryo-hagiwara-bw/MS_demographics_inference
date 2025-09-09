#!/usr/bin/env python3
"""
BigQuery query script for demographic inference project.
This script executes a query to find common_id records with 1000+ occurrences.
"""

from google.cloud import bigquery
import pandas as pd
import os

def main():
    # BigQueryクライアントの初期化
    # プロジェクトIDを明示的に指定
    project_id = "prd-analysis"
    try:
        client = bigquery.Client(project=project_id)
        print(f"BigQuery client initialized successfully for project: {project_id}")
    except Exception as e:
        print(f"Error initializing BigQuery client: {e}")
        print("Please ensure you have set up authentication for Google Cloud")
        return

    # 実行したいクエリ
    query = '''
    WITH
      -- ステップ1: 各common_idの行数を計算する
      counted_table AS (
        SELECT
          *,
          -- common_idごとにグループ化して、そのグループ内の行数を数える
          COUNT(common_id) OVER (PARTITION BY common_id) AS count_per_id
        FROM
           `prd-analysis.w_r_hagiwara.tokyo_demo_v3`
      )

    -- ステップ2: 計算した行数が1000以上のデータのみを抽出する
    SELECT
      common_id,
      arrive_ptime,
      depart_time,
      mesh,
      gender,
      age,
      name_of_prefectures,
      city_name,
      address,
      genre_name
    FROM
      counted_table
    WHERE
      count_per_id >= 1000
    LIMIT 10
    '''

    try:
        print("Executing BigQuery...")
        # クエリを実行
        result = client.query(query)
        
        # データフレームに変換
        df = result.to_dataframe()
        
        print(f"Query executed successfully. Retrieved {len(df)} rows.")
        print("\nFirst few rows:")
        print(df.head())
        
        # 結果をCSVファイルに保存（オプション）
        output_file = "bigquery_results.csv"
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        
        # 統計情報を表示
        print(f"\nDataFrame info:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        if not df.empty:
            print(f"\nUnique common_id count: {df['common_id'].nunique()}")
            print(f"Age range: {df['age'].min()} - {df['age'].max()}")
            print(f"Gender distribution:")
            print(df['gender'].value_counts())
        
    except Exception as e:
        print(f"Error executing query: {e}")
        return

if __name__ == "__main__":
    main()
