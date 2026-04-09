# ──────────────────────────────────────────────────────────────
    # Generate Comparison Report
    # ──────────────────────────────────────────────────────────────
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE ZERO STAGE COMPARISON")
    print(f"{'='*80}")
    
    # Create comparison table
    print("\nPerformance Metrics Comparison:")
    print("-" * 110)
    print(f"{'Stage':<8} {'Total Time':<12} {'Step Time':<12} {'Comm Time':<12} "
          f"{'Throughput':<12} {'VRAM':<10} {'Comm%':<8} {'Loss':<10}")
    print("-" * 110)
    
    for stage in sorted(all_results.keys()):
        r = all_results[stage]
        print(f"Stage {r['stage']:<3} "
              f"{r['total_training_time_s']:<12.1f} "
              f"{r['avg_step_time_ms']:<12.2f} "
              f"{r['avg_communication_time_ms']:<12.2f} "
              f"{r['avg_throughput_tok_s']:<12.1f} "
              f"{r['peak_gpu_memory_mb']:<10.0f} "
              f"{r['communication_overhead_pct']:<8.1f} "
              f"{r['final_loss']:<10.4f}")
    
    # Detailed analysis for report
    print(f"\n{'='*80}")
    print("DETAILED ANALYSIS FOR REPORT")
    print(f"{'='*80}")
    
    if 0 in all_results:
        baseline = all_results[0]
        
        print("\nCOMMUNICATION OVERHEAD ANALYSIS:")
        print("   Communication overhead increases significantly with higher ZeRO stages:")
        for stage in [1, 2, 3]:
            if stage in all_results:
                r = all_results[stage]
                overhead_increase = r['communication_overhead_pct'] - baseline['communication_overhead_pct']
                print(f"   Stage {stage}: {r['communication_overhead_pct']:.1f}% "
                      f"({overhead_increase:+.1f}% vs Stage 0)")
        
        print("\nTRADE-OFF ANALYSIS (Memory vs Runtime):")
        for stage in [1, 2, 3]:
            if stage in all_results:
                r = all_results[stage]
                memory_reduction = (baseline['peak_gpu_memory_mb'] - r['peak_gpu_memory_mb']) / baseline['peak_gpu_memory_mb'] * 100
                time_increase = (r['total_training_time_s'] - baseline['total_training_time_s']) / baseline['total_training_time_s'] * 100
                print(f"   Stage {stage}: {memory_reduction:.1f}% memory reduction, "
                      f"{time_increase:+.1f}% time increase")
        
        print("\nDIMINISHING RETURNS ANALYSIS:")
        prev_memory = baseline['peak_gpu_memory_mb']
        for stage in [1, 2, 3]:
            if stage in all_results:
                r = all_results[stage]
                marginal_reduction = (prev_memory - r['peak_gpu_memory_mb']) / prev_memory * 100
                print(f"   Stage {stage-1} → Stage {stage}: {marginal_reduction:.1f}% additional memory reduction")
                prev_memory = r['peak_gpu_memory_mb']
        
        print("\nBEST PERFORMING STAGE RECOMMENDATION:")
        # Multi-criteria decision analysis
        scores = {}
        for stage in all_results:
            r = all_results[stage]
            # Normalize metrics (higher is better for all)
            throughput_score = r['avg_throughput_tok_s'] / max([all_results[s]['avg_throughput_tok_s'] for s in all_results])
            memory_score = 1 - (r['peak_gpu_memory_mb'] / max([all_results[s]['peak_gpu_memory_mb'] for s in all_results]))
            time_score = 1 - (r['total_training_time_s'] / max([all_results[s]['total_training_time_s'] for s in all_results]))
            
            # Weighted score (throughput 40%, memory 35%, time 25%)
            scores[stage] = 0.4 * throughput_score + 0.35 * memory_score + 0.25 * time_score
        
        best_stage = max(scores, key=scores.get)
        print(f"   Based on weighted analysis (Throughput 40%, Memory 35%, Time 25%):")
        print(f"   ZeRO Stage {best_stage} achieves the best balance with score {scores[best_stage]:.3f}")
        
        if best_stage == 2:
            print("   → ZeRO Stage 2 provides the optimal trade-off: good memory savings")
            print("     without excessive communication overhead.")
        elif best_stage == 1:
            print("   → ZeRO Stage 1 offers modest memory savings with minimal overhead.")
        elif best_stage == 3:
            print("   → ZeRO Stage 3 provides maximum memory savings but at significant")
            print("     communication cost. Best when memory is the primary constraint.")
    
    # Save complete comparison
    comparison_file = os.path.join(base_output_dir, "complete_comparison.json")
    with open(comparison_file, "w") as f:
        json.dump(all_results, f, indent=4)
    
    print(f"\n✓ All results saved to: {base_output_dir}")
    print(f"✓ Comparison file: {comparison_file}")