from utils.plot_utils import generate_discount_plot
import argparse

parser = argparse.ArgumentParser(description="Generate Plots")
parser.add_argument('--log_folder', type=str, default=None, help="Log folder for logs generated by buttons_iql.py, enter in format 'logs/TIMESTAMP'")
parser.add_argument('--confidence_interval', '--ci', type=float, default=0.9, help='Confidence interval for plot. Default is 0.9.')
parser.add_argument('--smoothing', '--sm', type=int, default=50, help='Smoothing window size for graphs. Default is 50')
parser.add_argument('--gamma', '--g', type=float, default=0.99, help='discount factor')
args = parser.parse_args()

# logs/20240523-225901

if __name__ == "__main__":
    if not args.log_folder:
        raise Exception("Enter Valid Log Folder!!!!")

    # generate_plots(args.log_folder, args.smoothing, args.confidence_interval)
    generate_discount_plot(args.log_folder, args.smoothing, args.confidence_interval, args.gamma)
