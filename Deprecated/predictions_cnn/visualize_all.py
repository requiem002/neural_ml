import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_extended_window():
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'datasets'
    sub_dir = base_dir / 'submissions'
    
    datasets = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6']
    fig, axes = plt.subplots(len(datasets), 1, figsize=(16, 22), sharex=False)
    plt.subplots_adjust(hspace=0.6)

    # Increased window to 5000 samples (~0.2 seconds at 25kHz)
    start_sample = 50000 
    window_size = 5000 
    end_sample = start_sample + window_size

    for i, ds_name in enumerate(datasets):
        ax = axes[i]
        
        # 1. Load Raw Recording
        raw_path = data_dir / f'{ds_name}.mat'
        if not raw_path.exists(): continue
        raw_data = sio.loadmat(raw_path, squeeze_me=True)['d']
        
        # 2. Load System Results
        if ds_name == 'D1':
            res_path = raw_path # D1 uses its own labels for reference
        else:
            res_path = sub_dir / f'{ds_name}.mat'
            
        if not res_path.exists():
            ax.text(0.5, 0.5, f"{ds_name}: Missing Submission", ha='center')
            continue

        res_data = sio.loadmat(res_path, squeeze_me=True)
        indices = res_data['Index'] 
        classes = res_data['Class'] 

        # 3. Slice Data
        chunk = raw_data[start_sample:end_sample]
        time_axis = np.arange(start_sample, end_sample)
        ax.plot(time_axis, chunk, color='black', linewidth=0.5, alpha=0.7)
        
        # 4. Filter detections to window
        mask = (indices >= start_sample) & (indices <= end_sample)
        local_indices = indices[mask]
        local_classes = classes[mask]

        # 5. Visual Markers
        for idx, cls in zip(local_indices, local_classes):
            ax.axvline(x=idx-1, color='magenta', linestyle=':', alpha=0.8)
            ax.text(idx-1, ax.get_ylim()[1]*0.7, f"C{cls}", 
                    color='magenta', weight='bold', ha='center', fontsize=9)

        # 6. Formatting based on brief details
        ax.set_title(f"Dataset: {ds_name} | Recordings from Bipolar Electrode [cite: 12]", loc='left', fontweight='bold')
        ax.set_ylabel("Amplitude (mV) [cite: 22]")
        ax.grid(True, alpha=0.2)

    plt.xlabel("Sample Index (25 kHz Sampling Rate) [cite: 21]")
    plt.show()

if __name__ == "__main__":
    visualize_extended_window()