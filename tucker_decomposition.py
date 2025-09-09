#!/usr/bin/env python3
"""
メッシュ粒度を削減したTucker分解スクリプト
メッシュの右の一桁を削ってテンソルサイズを大幅に削減
"""

import tensorly as tl
import numpy as np
import pandas as pd
import json

def load_and_reduce_mesh_granularity():
    """
    データを読み込んでメッシュ粒度を削減
    """
    print("=== メッシュ粒度削減によるデータ前処理 ===")
    
    # 元のBigQueryデータを読み込み
    df = pd.read_csv('bigquery_results.csv')
    
    print(f"元のデータ行数: {len(df)}")
    print(f"元のユニークなmesh数: {df['mesh'].nunique()}")
    
    # メッシュの右の一桁を削除（例: 53394621 → 5339462）
    df['mesh_reduced'] = df['mesh'] // 10
    
    print(f"削減後のユニークなmesh数: {df['mesh_reduced'].nunique()}")
    print(f"メッシュ削減率: {df['mesh_reduced'].nunique() / df['mesh'].nunique():.4f}")
    
    # テンソル構築用のデータを準備
    tensor_df = df[['common_id', 'arrive_ptime', 'mesh_reduced', 'genre_name']].copy()
    
    # 時間的特徴量を作成
    tensor_df['arrive_ptime'] = pd.to_datetime(tensor_df['arrive_ptime'])
    tensor_df['arrive_ptime_jst'] = tensor_df['arrive_ptime'].dt.tz_convert('Asia/Tokyo')
    
    day_of_week = tensor_df['arrive_ptime_jst'].dt.dayofweek
    is_weekday = day_of_week < 5
    hour = tensor_df['arrive_ptime_jst'].dt.hour
    
    tensor_df['time_bin'] = np.where(
        is_weekday,
        'weekday_' + hour.astype(str).str.zfill(2),
        'weekend_' + hour.astype(str).str.zfill(2)
    )
    
    # 各軸のユニークIDを作成
    unique_users = tensor_df['common_id'].unique()
    user_map = {user: i for i, user in enumerate(unique_users)}
    tensor_df['user_id'] = tensor_df['common_id'].map(user_map)
    
    unique_genres = tensor_df['genre_name'].unique()
    genre_map = {genre: i for i, genre in enumerate(unique_genres)}
    tensor_df['genre_id'] = tensor_df['genre_name'].map(genre_map)
    
    unique_meshes = tensor_df['mesh_reduced'].unique()
    mesh_map = {mesh: i for i, mesh in enumerate(unique_meshes)}
    tensor_df['mesh_id'] = tensor_df['mesh_reduced'].map(mesh_map)
    
    unique_time_bins = tensor_df['time_bin'].unique()
    time_bin_map = {tb: i for i, tb in enumerate(unique_time_bins)}
    tensor_df['time_bin_id'] = tensor_df['time_bin'].map(time_bin_map)
    
    # コンテキストIDを作成
    num_time_bins = len(unique_time_bins)
    tensor_df['context_id'] = tensor_df['mesh_id'] * num_time_bins + tensor_df['time_bin_id']
    
    # テンソルを構築
    grouped = tensor_df.groupby(['user_id', 'context_id', 'genre_id']).size()
    tensor_coords = np.array(grouped.index.to_list())
    tensor_values = grouped.values
    
    num_users = len(unique_users)
    num_context_dims = len(unique_meshes) * num_time_bins
    num_genre_dims = len(unique_genres)
    tensor_shape = (num_users, num_context_dims, num_genre_dims)
    
    print(f"\n=== テンソル構築結果 ===")
    print(f"ユーザー数: {num_users}")
    print(f"メッシュ数（削減後）: {len(unique_meshes)}")
    print(f"時間ビン数: {num_time_bins}")
    print(f"ジャンル数: {len(unique_genres)}")
    print(f"テンソル形状: {tensor_shape}")
    print(f"非ゼロ要素数: {len(tensor_values)}")
    print(f"スパース性: {len(tensor_values) / np.prod(tensor_shape):.8f}")
    
    # メモリ使用量を計算
    memory_usage = np.prod(tensor_shape) * 8 / (1024**3)  # GB
    print(f"推定メモリ使用量: {memory_usage:.4f} GB")
    
    return tensor_coords, tensor_values, tensor_shape, user_map, genre_map

def create_dense_tensor(tensor_coords, tensor_values, tensor_shape):
    """
    密テンソルを作成
    """
    print("\n=== 密テンソルの作成 ===")
    
    dense_tensor = np.zeros(tensor_shape)
    for i, coord in enumerate(tensor_coords):
        user_id, context_id, genre_id = coord
        value = tensor_values[i]
        dense_tensor[user_id, context_id, genre_id] = value
    
    print(f"密テンソルの形状: {dense_tensor.shape}")
    print(f"非ゼロ要素数: {np.count_nonzero(dense_tensor)}")
    print(f"スパース性: {np.count_nonzero(dense_tensor) / np.prod(tensor_shape):.8f}")
    
    # 実際のメモリ使用量
    actual_memory = dense_tensor.nbytes / (1024**3)  # GB
    print(f"実際のメモリ使用量: {actual_memory:.4f} GB")
    
    return dense_tensor

def perform_standard_tucker_decomposition(dense_tensor, tensor_shape):
    """
    スタンダードなTucker分解を実行
    """
    print("\n=== スタンダードなTucker分解 ===")
    
    # ランクを設定（テンソルサイズに応じて調整）
    ranks = [
        min(20, tensor_shape[0]),     # ユーザー次元
        min(100, tensor_shape[1]),    # コンテキスト次元
        min(50, tensor_shape[2])      # 場所の知識次元
    ]
    
    print(f"設定ランク: {ranks}")
    print(f"元の次元数: {tensor_shape}")
    print(f"圧縮率: {np.prod(ranks) / np.prod(tensor_shape):.8f}")
    
    print("\nTucker分解を実行中...")
    try:
        core, factors = tl.decomposition.tucker(
            dense_tensor, 
            rank=ranks, 
            init='random', 
            verbose=1,
            n_iter_max=30
        )
        
        print("\n--- スタンダードなTucker分解完了 ---")
        print(f"コアテンソルの形状: {core.shape}")
        
        # 各因子行列の形状を表示
        for i, (factor, rank) in enumerate(zip(factors, ranks)):
            print(f"因子行列[{i}]の形状: {factor.shape} (ランク: {rank})")
        
        return core, factors, ranks
        
    except Exception as e:
        print(f"Tucker分解でエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def analyze_standard_results(core, factors, ranks, tensor_shape, user_map, genre_map):
    """
    スタンダードな分解結果を分析
    """
    print("\n=== スタンダードな分解結果の分析 ===")
    
    if core is None:
        print("分解に失敗しました。")
        return
    
    # ユーザー特徴量行列（factors[0]）
    user_features = factors[0]
    print(f"ユーザー特徴量行列の形状: {user_features.shape}")
    print("最初の10ユーザーの特徴ベクトル:")
    
    for i in range(min(10, len(user_features))):
        user_id = list(user_map.keys())[i]
        print(f"  ユーザー {user_id}: {user_features[i, :]}")
    
    # 特徴量の統計情報
    print(f"\n特徴量の統計情報:")
    print(f"平均値: {np.mean(user_features):.6f}")
    print(f"標準偏差: {np.std(user_features):.6f}")
    print(f"最小値: {np.min(user_features):.6f}")
    print(f"最大値: {np.max(user_features):.6f}")
    
    # 各ユーザーの特徴量のノルム
    user_norms = np.linalg.norm(user_features, axis=1)
    print(f"\nユーザー特徴量のノルム統計:")
    print(f"平均ノルム: {np.mean(user_norms):.6f}")
    print(f"ノルムの標準偏差: {np.std(user_norms):.6f}")
    
    # ユーザー間の類似度を計算（最初の10ユーザー）
    print(f"\nユーザー間の類似度（コサイン類似度、最初の10ユーザー）:")
    for i in range(min(10, len(user_features))):
        for j in range(i+1, min(10, len(user_features))):
            similarity = np.dot(user_features[i], user_features[j]) / (np.linalg.norm(user_features[i]) * np.linalg.norm(user_features[j]))
            user_i_id = list(user_map.keys())[i]
            user_j_id = list(user_map.keys())[j]
            print(f"  {user_i_id} vs {user_j_id}: {similarity:.4f}")
    
    return user_features

def save_standard_results(core, factors, ranks, user_features, user_map, genre_map):
    """
    スタンダードな結果を保存
    """
    print("\n=== スタンダードな結果の保存 ===")
    
    if core is None:
        print("保存する結果がありません。")
        return
    
    # コアテンソルと因子行列を個別に保存
    np.save('tucker_core_standard.npy', core)
    for i, factor in enumerate(factors):
        np.save(f'tucker_factor_{i}_standard.npy', factor)
    
    # ユーザー特徴量を個別に保存
    np.save('user_features_standard.npy', user_features)
    
    # マッピング情報を保存（キーを文字列に変換）
    user_map_str = {str(k): v for k, v in user_map.items()}
    genre_map_str = {str(k): v for k, v in genre_map.items()}
    
    with open('user_mapping_standard.json', 'w') as f:
        json.dump(user_map_str, f, ensure_ascii=False, indent=2)
    
    with open('genre_mapping_standard.json', 'w') as f:
        json.dump(genre_map_str, f, ensure_ascii=False, indent=2)
    
    # ランク情報をJSONで保存
    with open('tucker_ranks_standard.json', 'w') as f:
        json.dump(ranks, f, indent=2)
    
    print("保存されたファイル:")
    print("- tucker_core_standard.npy: コアテンソル")
    print("- tucker_factor_0_standard.npy: ユーザー因子行列")
    print("- tucker_factor_1_standard.npy: コンテキスト因子行列")
    print("- tucker_factor_2_standard.npy: ジャンル因子行列")
    print("- user_features_standard.npy: ユーザー特徴量行列")
    print("- user_mapping_standard.json: ユーザーIDマッピング")
    print("- genre_mapping_standard.json: ジャンルIDマッピング")
    print("- tucker_ranks_standard.json: ランク情報")

def main():
    """
    メイン実行関数
    """
    print("=== メッシュ粒度削減によるスタンダードなTucker分解 ===")
    
    # 1. データを読み込んでメッシュ粒度を削減
    tensor_coords, tensor_values, tensor_shape, user_map, genre_map = load_and_reduce_mesh_granularity()
    
    # 2. 密テンソルを作成
    dense_tensor = create_dense_tensor(tensor_coords, tensor_values, tensor_shape)
    
    # 3. スタンダードなTucker分解を実行
    core, factors, ranks = perform_standard_tucker_decomposition(dense_tensor, tensor_shape)
    
    # 4. 結果を分析
    user_features = analyze_standard_results(core, factors, ranks, tensor_shape, user_map, genre_map)
    
    # 5. 結果を保存
    save_standard_results(core, factors, ranks, user_features, user_map, genre_map)
    
    if core is not None:
        print("\n=== 完了 ===")
        print("メッシュ粒度削減によるスタンダードなTucker分解が正常に完了しました。")
        print("テンソルサイズを大幅に削減してメモリ効率的にユーザー特徴量が抽出されました。")
    else:
        print("\n=== 失敗 ===")
        print("スタンダードなTucker分解でエラーが発生しました。")

if __name__ == "__main__":
    main()
