import { cn } from "@/lib/utils"

interface SkeletonProps extends React.HTMLAttributes<HTMLDivElement> {
  count?: number;
}

function Skeleton({ className, count, ...props }: SkeletonProps) {
  return (
    <>
      {count &&
        Array.from({ length: count }).map((_, i) => (
          <div
            key={i}
            className={cn("animate-pulse rounded-md bg-muted", className)}
            {...props}
          />
        ))}
      {!count && <div className={cn("animate-pulse rounded-md bg-muted", className)} {...props} />}
    </>
  );
}

export { Skeleton };
