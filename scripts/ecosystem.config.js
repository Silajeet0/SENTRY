// pm2 process definitions for SENTRY's public-facing API.
//
//   pm2 start scripts/ecosystem.config.js
//   pm2 save                 # persist the process list
//   pm2 startup              # (one-time) install the launchd boot hook
//
// Useful commands once running: `pm2 status`, `pm2 logs sentry-api`,
// `pm2 logs sentry-watcher` — handy since nobody needs to be physically
// at the Mac to check on either process anymore.

module.exports = {
  apps: [
    {
      name: "sentry-api",
      script: "orchestrator_api.py",
      interpreter: "python3",       // point at the venv's python3 if using one
      cwd: __dirname + "/..",
      autorestart: true,
      max_restarts: 10,
    },
    {
      name: "sentry-watcher",
      script: "./scripts/watch_and_restart.sh",
      cwd: __dirname + "/..",
      autorestart: true,
      max_restarts: 10,
    },
    {
      // Deliberately its own pm2 app, separate from sentry-api — the
      // watcher restarts sentry-api on every new commit, and this must
      // NOT be restarted along with it, or the public URL would rotate
      // on every deploy. This one only restarts if cloudflared itself
      // dies (crash, Mac reboot, long network outage).
      name: "sentry-tunnel",
      script: "cloudflared",
      args: "tunnel --url http://localhost:8091",
      autorestart: true,
      max_restarts: 10,
    },
  ],
};
