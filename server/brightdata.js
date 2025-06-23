const brightDataTriggerUrl = 'https://api.brightdata.com/datasets/v3/trigger';

export const triggerYoutubeVideoScrape = async (url) => {
  // Build the public webhook URL at call-time (makes local tunnelling easier)
  const webhookUrl = `${process.env.API_URL}/webhook`;

  const data = JSON.stringify([{ url, country: '' }]);

  try {
    const response = await fetch(
      `${brightDataTriggerUrl}?dataset_id=gd_lk56epmy2i5g7lzu0k&endpoint=${encodeURIComponent(
        webhookUrl
      )}&format=json&uncompressed_webhook=true&include_errors=true`,
      {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${process.env.BRIGHTDATA_API_KEY}`,
          'Content-Type': 'application/json',
        },
        body: data,
      }
    );

    if (!response.ok) {
      throw new Error(`BrightData API error: ${response.status}`);
    }

    const result = await response.json();
    console.log('BrightData response:', result);

    return result.snapshot_id;
  } catch (error) {
    console.error('Error triggering YouTube video scrape:', error);
    throw error;
  }
};
