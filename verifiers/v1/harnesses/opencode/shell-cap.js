// Copy stdin to stdout as it arrives, up to argv[2] bytes; past that, say so once and
// drop the rest. Unbuffered, so a background job that keeps the pipe open does not
// hold back what came before it.
const limit = Number(process.argv[2]);
let seen = 0;
process.stdout.on("error", () => process.exit(0));
process.stdin.on("data", (chunk) => {
  if (seen < limit) {
    const room = limit - seen;
    process.stdout.write(chunk.length > room ? chunk.subarray(0, room) : chunk);
  }
  if (seen <= limit && seen + chunk.length > limit) {
    process.stdout.write(`\n[output past ${limit} bytes dropped]\n`);
  }
  seen += chunk.length;
});
