module.exports = [
  [
    "@docusaurus/preset-classic",
    {
      docs: false,
      blog: false,
      theme: {
        customCss: require.resolve("../src/css/custom.css"),
      },
    },
  ],
];
