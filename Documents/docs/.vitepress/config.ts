import { defineConfig } from 'vitepress'

function resolveBase() {
  const repo = process.env.GITHUB_REPOSITORY?.split('/')[1]
  if (!process.env.GITHUB_ACTIONS || !repo) {
    return '/'
  }
  return repo.endsWith('.github.io') ? '/' : `/${repo}/`
}

export default defineConfig({
  base: resolveBase(),
  lang: 'zh-CN',
  title: 'shmtu-cas-ocr-model',
  description: '上海海事大学 CAS 验证码 OCR 识别模型训练项目',
  cleanUrls: true,
  lastUpdated: true,
  ignoreDeadLinks: true,
  themeConfig: {
    nav: [
      { text: '概览', link: '/' },
      { text: '使用说明', link: '/usage/v2-quickstart' },
      { text: 'API 服务器', link: '/api-server/overview' },
      { text: '论文', link: '/paper/abstract' },
    ],
    sidebar: [
      {
        text: '概览',
        items: [
          { text: '项目首页', link: '/' },
        ],
      },
      {
        text: '使用说明 — V2 (当前版本)',
        items: [
          { text: '快速开始', link: '/usage/v2-quickstart' },
          { text: '训练配置', link: '/usage/v2-training-config' },
          { text: '评估与推理', link: '/usage/v2-eval-inference' },
          { text: '模型导出 (ONNX/NCNN)', link: '/usage/v2-export' },
          { text: '数据采集', link: '/usage/v2-data-collection' },
        ],
      },
      {
        text: '使用说明 — V1 (旧版)',
        items: [
          { text: 'V1 架构与使用', link: '/usage/v1-overview' },
        ],
      },
      {
        text: 'API 服务器',
        items: [
          { text: '概览与启动', link: '/api-server/overview' },
          { text: 'HTTP 接口', link: '/api-server/http-api' },
          { text: 'TCP 协议', link: '/api-server/tcp-protocol' },
        ],
      },
      {
        text: '论文',
        items: [
          { text: '摘要', link: '/paper/abstract' },
          { text: '1. 引言', link: '/paper/introduction' },
          { text: '2. 数据采集', link: '/paper/data-collection' },
          { text: '3. 模型架构', link: '/paper/model-architecture' },
          { text: '4. 数据增强与损失函数', link: '/paper/augmentation-loss' },
          { text: '5. 实验', link: '/paper/experiments' },
          { text: '6. 结论', link: '/paper/conclusion' },
        ],
      },
    ],
    outline: [2, 3],
    search: {
      provider: 'local',
    },
    footer: {
      message: 'shmtu-cas-ocr-model Docs',
      copyright: 'Copyright © shmtu-cas-ocr-model',
    },
  },
})
